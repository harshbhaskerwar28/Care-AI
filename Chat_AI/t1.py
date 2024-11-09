import os
import io
import asyncio
import aiohttp
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time
import json

# Load environment variables
load_dotenv()

# [Previous dataclass definitions and DocumentProcessor class remain the same]
# ... [Keep all the code until the HealthcareAgent class]

class HealthcareAgent:
    """Healthcare agent with concise response generation"""
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.chat_history = []
        self.doc_processor = DocumentProcessor()
        self._initialize_prompts()
        self.agents = self._initialize_agents()

    def _initialize_prompts(self):
        """Initialize prompts optimized for concise responses"""
        self.prompts = {
            'main_agent': """You are a healthcare coordinator AI. Be direct and concise.
Context: {context}
Query: {query}
Chat History: {chat_history}

Provide a brief response with:
1. Key medical concepts (2-3 points)
2. Necessary specialist consultations
3. Quick initial assessment
Limit response to 3-4 sentences.""",

            'diagnosis_agent': """You are a medical diagnosis specialist. Be concise.
Context: {context}
Query: {query}
Chat History: {chat_history}

Provide brief:
1. Key symptoms identified
2. Top 2-3 potential conditions
3. Immediate next steps
Limit to 3-4 key points.""",

            'treatment_agent': """You are a treatment specialist. Be direct.
Context: {context}
Query: {query}
Chat History: {chat_history}

Provide only:
1. Top 1-2 treatment options
2. Key lifestyle changes
3. Critical warning signs
Keep response under 100 words.""",

            'research_agent': """You are a medical research specialist. Be brief.
Context: {context}
Query: {query}
Chat History: {chat_history}

Provide only:
1. Most relevant research finding
2. Key clinical guideline
3. Primary recommendation
Limit to 2-3 sentences.""",

            'synthesis_agent': """You are a medical information synthesizer. Be concise.
Context: {context}
Query: {query}
Chat History: {chat_history}
Agent Responses: {agent_responses}

Provide a clear, concise summary:
1. Main recommendation
2. Key action items
3. Important warnings (if any)

Keep the final response under 150 words and focus on practical next steps.
For simple queries (like greetings), respond in one short sentence."""
        }

    # [Rest of the HealthcareAgent methods remain the same]

def setup_streamlit_ui():
    """Setup Streamlit UI with dark sidebar"""
    # [Previous UI setup code remains the same]
    
    # Add styles for more compact messages
    st.markdown("""
        <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            border: 1px solid #dee2e6;
            background-color: black;
            font-size: 0.95rem;
        }
        .agent-card {
            padding: 0.8rem;
            margin-bottom: 0.5rem;
        }
        .metadata-section {
            font-size: 0.75rem;
            margin-top: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    """Main application with optimized response handling"""
    setup_streamlit_ui()
    
    # [Previous session state initialization code remains the same]
    
    # Main content area with modified title
    st.title("🏥 Healthcare AI Assistant")
    st.markdown("""
        Upload medical documents and ask questions for concise, practical answers.
    """)
    
    # [Rest of the main function remains the same]

if __name__ == "__main__":
    main()
