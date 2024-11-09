import os
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time

load_dotenv()

st.set_page_config(
    page_title="Health Report Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        .workflow-container {
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
            margin: 10px;
        }
        
        .workflow-step {
            background-color: #1E1E1E;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 3px solid #00CA51;
        }
        
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        
        .user-message {
            background-color: #262730;
            border-left: 5px solid #00CA51;
        }
        
        .assistant-message {
            background-color: #1E1E1E;
            border-left: 5px solid #0078FF;
        }
        
        .agent-status {
            background-color: #1E1E1E;
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .agent-status.idle {
            border-left: 4px solid #808080;
        }
        
        .agent-status.processing {
            border-left: 4px solid #FFA500;
            animation: pulse 2s infinite;
        }
        
        .agent-status.complete {
            border-left: 4px solid #00CA51;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.6; }
            100% { opacity: 1; }
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #262730;
            border-radius: 4px;
            padding: 8px 16px;
            color: #FAFAFA;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #404040;
        }
        
        .stProgress > div > div {
            background-color: #00CA51;
        }
        
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
""", unsafe_allow_html=True)

@dataclass
class AgentResponse:
    """Structure for agent responses"""
    agent_name: str
    content: str
    confidence: float
    processing_time: float

class HealthReportAnalyzer:
    # Your existing HealthReportAnalyzer implementation here
    pass

def display_workflow():
    """Display the analysis workflow with updated styling"""
    with st.container():
        st.markdown("""
            <div class="workflow-container">
                <h3>How It Works</h3>
                <div class="workflow-step">
                    1. Upload Your Health Report (PDF/TXT)
                </div>
                <div class="workflow-step">
                    2. AI Agents Analyze Your Report
                </div>
                <div class="workflow-step">
                    3. Get Comprehensive Analysis & Insights
                </div>
                <div class="workflow-step">
                    4. Chat with AI About Your Results
                </div>
            </div>
        """, unsafe_allow_html=True)

def display_agent_status():
    """Display agent status in sidebar with improved styling and loading indicators"""
    st.sidebar.markdown("### 🤖 Agent Status")
    
    agents = {
        'Document Processor': '📄',
        'Positive Analyzer': '✨',
        'Risk Assessor': '⚠️',
        'Summary Generator': '📝',
        'Recommendation Engine': '💡'
    }
    
    for agent_name, icon in agents.items():
        status_class = "idle"
        status_text = "Idle"
        loading_spinner = ""
        
        if hasattr(st.session_state, 'processing_agent'):
            if st.session_state.processing_agent == agent_name:
                status_class = "processing"
                status_text = "Processing"
                loading_spinner = '<div class="loading-spinner"></div>'
            elif agent_name in st.session_state.completed_agents:
                status_class = "complete"
                status_text = "Complete"
        
        st.sidebar.markdown(f"""
            <div class="agent-status {status_class}">
                <div>{icon} {agent_name}</div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    {status_text}
                    {loading_spinner}
                </div>
            </div>
        """, unsafe_allow_html=True)

def handle_chat_input():
    """Handle chat input and response with improved flow control"""
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""
    if "processing_message" not in st.session_state:
        st.session_state.processing_message = False
    
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("Ask a question about your report:", 
                                  key="chat_input_field",
                                  disabled=st.session_state.processing_message)
    with col2:
        send_button = st.button("Send", disabled=st.session_state.processing_message)
    
    if send_button and user_input and not st.session_state.processing_message:
        st.session_state.processing_message = True
        
        if st.session_state.report_text:
            with st.spinner("Processing your question..."):
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                
                response = asyncio.run(
                    st.session_state.analyzer.generate_chat_response(
                        user_input,
                        st.session_state.report_text
                    )
                )
                
                st.session_state.chat_history.append(AIMessage(content=response))
                
                st.session_state.chat_input = ""
                st.session_state.processing_message = False
                st.rerun()

def main():
    """Main application with enhanced UI and fixed chat loop"""
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = HealthReportAnalyzer()
    if 'report_results' not in st.session_state:
        st.session_state.report_results = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'report_text' not in st.session_state:
        st.session_state.report_text = None
    if 'processing_agent' not in st.session_state:
        st.session_state.processing_agent = None
    if 'completed_agents' not in st.session_state:
        st.session_state.completed_agents = set()
    
    # Sidebar
    with st.sidebar:
        st.title("🏥 Health Report Analyzer")
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Upload your health report",
            type=['pdf', 'txt'],
            key='file_uploader'
        )
        
        analyze_button = st.button(
            "🔍 Analyze Report",
            key='analyze_btn',
            disabled=st.session_state.processing_agent is not None
        )
        
        if uploaded_file and analyze_button:
            try:
                if uploaded_file.type == "application/pdf":
                    pdf_reader = PdfReader(uploaded_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                else:
                    text = uploaded_file.getvalue().decode()
                
                st.session_state.report_text = text
                st.session_state.processing_agent = None
                st.session_state.completed_agents = set()
                
                st.session_state.report_results = asyncio.run(
                    st.session_state.analyzer.analyze_report(text)
                )
                
                st.success("Analysis complete!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing report: {str(e)}")
        
        display_agent_status()
    
    # Main content area
    st.title("Health Report Analysis")
    
    if not st.session_state.report_results:
        display_workflow()
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "✅ Positive Findings",
            "⚠️ Areas of Concern",
            "📊 Full Report",
            "💬 Chat Assistant"
        ])
        
        with tab1:
            if 'positive_analyzer' in st.session_state.report_results:
                result = st.session_state.report_results['positive_analyzer']
                st.markdown(result.content)
        
        with tab2:
            if 'negative_analyzer' in st.session_state.report_results:
                result = st.session_state.report_results['negative_analyzer']
                st.markdown(result.content)
        
        with tab3:
            if 'document_processor' in st.session_state.report_results:
                st.subheader("Document Analysis")
                st.markdown(st.session_state.report_results['document_processor'].content)
            
            if 'summary_agent' in st.session_state.report_results:
                st.subheader("Summary")
                st.markdown(st.session_state.report_results['summary_agent'].content)
            
            if 'recommendation_agent' in st.session_state.report_results:
                st.subheader("Recommendations")
                st.markdown(st.session_state.report_results['recommendation_agent'].content)
        
        with tab4:
            for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>You:</strong> {message.content}
                        </div>
                    """, unsafe_allow_html=True)
                elif isinstance(message, AIMessage):
                    st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>Assistant:</strong> {message.content}
                        </div>
                    """, unsafe_allow_html=True)
            
            handle_chat_input()

if __name__ == "__main__":
    main()
