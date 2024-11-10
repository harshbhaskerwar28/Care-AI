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

@dataclass
class ProcessedDocument:
    """Structure for processed document information"""
    filename: str
    content: str
    chunks: List[str]
    total_chars: int
    doc_type: str
    summary: str = ""

@dataclass
class AgentResponse:
    """Structure for storing agent responses"""
    agent_name: str
    content: str
    confidence: float
    metadata: Dict = None
    processing_time: float = 0.0

class DocumentProcessor:
    """Document processing with vector store integration"""
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.processed_documents: List[ProcessedDocument] = []
        self._initialize_embeddings()
        self.vector_store = None

    def _initialize_embeddings(self):
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {str(e)}")
            raise

    async def process_file(self, file, progress_callback) -> ProcessedDocument:
        try:
            progress_callback(0.2, f"Processing {file.name}")
            
            content = ""
            if file.type == "application/pdf":
                content = await self.process_pdf(file)
                doc_type = "PDF"
            elif file.type.startswith("image/"):
                content = await self.process_image(file)
                doc_type = "Image"
            else:
                raise ValueError(f"Unsupported file type: {file.type}")

            progress_callback(0.4, "Splitting content into chunks")
            chunks = self.text_splitter.split_text(content)
            
            progress_callback(0.6, "Generating document summary")
            summary = await self._generate_summary(content[:1000])
            
            processed_doc = ProcessedDocument(
                filename=file.name,
                content=content,
                chunks=chunks,
                total_chars=len(content),
                doc_type=doc_type,
                summary=summary
            )
            
            self.processed_documents.append(processed_doc)
            return processed_doc

        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return None

    async def process_pdf(self, pdf_file) -> str:
        text = ""
        try:
            bytes_data = pdf_file.read()
            pdf_reader = PdfReader(io.BytesIO(bytes_data))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"PDF processing error: {str(e)}")

    async def process_image(self, image_file) -> str:
        try:
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode not in ('L', 'RGB'):
                image = image.convert('RGB')
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            raise Exception(f"Image processing error: {str(e)}")

    async def _generate_summary(self, text: str) -> str:
        return f"{text[:200]}..."

    def get_relevant_context(self, query: str, num_chunks: int = 3) -> str:
        """Get relevant context from vector store"""
        try:
            if self.vector_store is None or not query.strip():
                return ""
            results = self.vector_store.similarity_search(query, k=num_chunks)
            return "\n\n".join([doc.page_content for doc in results])
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
            return ""

class HealthcareAgent:
    """Healthcare agent with improved response handling"""
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
        """Initialize concise agent prompts"""
        self.prompts = {
            'main_agent': """You are a healthcare coordinator AI. Be concise and direct. For simple greetings or non-medical queries, respond briefly and naturally.
For medical queries, analyze the context and query to determine if specialist consultation is needed.

Context: {context}
Query: {query}
Chat History: {chat_history}

Respond appropriately to the query type:
1. For greetings/simple queries: Give a brief, friendly response
2. For medical queries: Provide a concise initial assessment and determine if specialist input is needed""",

            'synthesis_agent': """You are a medical information synthesizer. Keep responses focused and concise.
Context: {context}
Query: {query}
Chat History: {chat_history}
Agent Responses: {agent_responses}

Provide a clear, concise response that:
1. Addresses the query directly
2. Includes only relevant information
3. Uses simple language
4. Maintains appropriate length for query complexity"""
        }

        # Add other specialist agent prompts similarly...

    def _initialize_agents(self):
        return {
            name: ChatPromptTemplate.from_messages([
                ("system", prompt),
                ("human", "{input}")
            ]) | self.llm | StrOutputParser()
            for name, prompt in self.prompts.items()
        }

    def _format_chat_history(self) -> str:
        formatted = []
        for msg in self.chat_history[-3:]:  # Last 3 messages only
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    async def process_query(self, query: str, status_callback) -> Dict[str, AgentResponse]:
        """Process query with improved response handling"""
        try:
            # For simple greetings or non-medical queries, use main agent only
            if self._is_simple_query(query):
                response = await self._get_agent_response(
                    'main_agent',
                    query,
                    "",  # No context needed for simple queries
                    self._format_chat_history()
                )
                return {'main_agent': response}

            # For medical queries, use full agent system
            context = self.doc_processor.get_relevant_context(query)
            responses = await self._process_medical_query(query, context, status_callback)
            return responses

        except Exception as e:
            status_callback('main_agent', 'error', 0, str(e))
            raise

    def _is_simple_query(self, query: str) -> bool:
        """Determine if query is a simple greeting or non-medical query"""
        simple_patterns = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 
            'good evening', 'how are you', 'thanks', 'thank you'
        ]
        return any(query.lower().strip().startswith(pattern) for pattern in simple_patterns)

    async def _process_medical_query(self, query: str, context: str, status_callback) -> Dict[str, AgentResponse]:
        """Process medical queries with full agent system"""
        responses = {}
        chat_history = self._format_chat_history()

        # Get main agent response
        status_callback('main_agent', 'working', 0.2, "Analyzing query")
        main_response = await self._get_agent_response(
            'main_agent',
            query,
            context,
            chat_history
        )
        responses['main_agent'] = main_response

        # Process through specialist agents if needed
        if self._needs_specialist_consultation(main_response.content):
            specialist_responses = await self._get_specialist_responses(
                query,
                context,
                chat_history,
                status_callback
            )
            responses.update(specialist_responses)

        # Synthesize final response
        status_callback('synthesis_agent', 'working', 0.8, "Creating final response")
        final_response = await self._synthesize_responses(
            query,
            context,
            chat_history,
            responses
        )
        responses['synthesis_agent'] = final_response

        return responses

    def _needs_specialist_consultation(self, main_response: str) -> bool:
        """Determine if specialist consultation is needed based on main agent response"""
        medical_indicators = [
            'symptom', 'condition', 'treatment', 'diagnosis', 'medical',
            'health', 'pain', 'doctor', 'medicine', 'disease'
        ]
        return any(indicator in main_response.lower() for indicator in medical_indicators)



class HealthcareAgent:
    """Enhanced healthcare agent with simplified document storage"""
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
        """Initialize enhanced agent prompts"""
        self.prompts = {
            'main_agent': """You are a healthcare coordinator AI. Analyze the following:
Context: {context}
Query: {query}
Chat History: {chat_history}

Your task:
1. Understand the main medical query
2. Identify key medical concepts
3. Determine which specialist agents to consult
4. Create a structured analysis plan

Provide a detailed response including:
1. Query understanding
2. Key medical concepts identified
3. Suggested specialist consultations
4. Initial assessment
5. Questions for clarification (if needed)""",

            'diagnosis_agent': """You are a medical diagnosis specialist. Analyze:
Context: {context}
Query: {query}
Chat History: {chat_history}

Focus on:
1. Symptom analysis
2. Potential conditions
3. Risk assessment
4. Diagnostic considerations
5. Medical history relevance

Provide a structured response with:
1. Detailed symptom analysis
2. Differential diagnosis
3. Risk factors identified
4. Recommended diagnostic steps
5. Urgency assessment""",

            'treatment_agent': """You are a treatment specialist. Consider:
Context: {context}
Query: {query}
Chat History: {chat_history}

Analyze:
1. Treatment options
2. Evidence-based approaches
3. Risk-benefit analysis
4. Treatment timeline
5. Monitoring requirements

Provide recommendations for:
1. Immediate steps
2. Treatment options
3. Lifestyle modifications
4. Follow-up care
5. Warning signs to watch""",

            'research_agent': """You are a medical research specialist. Research:
Context: {context}
Query: {query}
Chat History: {chat_history}

Focus on:
1. Recent research findings
2. Clinical guidelines
3. Treatment effectiveness
4. Safety considerations
5. Emerging approaches

Provide analysis of:
1. Current medical evidence
2. Research quality
3. Clinical applications
4. Future directions
5. Knowledge gaps""",

            'synthesis_agent': """You are a medical information synthesizer. Integrate:
Context: {context}
Query: {query}
Chat History: {chat_history}
Agent Responses: {agent_responses}

Create a comprehensive response:
1. Summarize key findings
2. Integrate specialist insights
3. Provide clear recommendations
4. Address uncertainties
5. Suggest next steps

Ensure the response is:
1. Clear and actionable
2. Evidence-based
3. Patient-centered
4. Safety-conscious
5. Well-structured"""
        }

    def _initialize_agents(self):
        """Initialize enhanced agent system"""
        return {
            name: ChatPromptTemplate.from_messages([
                ("system", prompt),
                ("human", "{input}")
            ]) | self.llm | StrOutputParser()
            for name, prompt in self.prompts.items()
        }

    def _format_chat_history(self) -> str:
        """Format chat history for context"""
        formatted = []
        for msg in self.chat_history[-5:]:  # Last 5 messages
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    async def process_documents(self, files, status_callback) -> bool:
        """Process documents with detailed status updates"""
        try:
            processed_docs = []
            
            for idx, file in enumerate(files):
                doc = await self.doc_processor.process_file(
                    file,
                    lambda p, m: status_callback(
                        'document_processor',
                        'working',
                        (idx / len(files)) + (p / len(files)),
                        m
                    )
                )
                if doc:
                    processed_docs.append(doc)

            if processed_docs:
                success = await self.doc_processor.update_document_store(
                    processed_docs,
                    lambda p, m: status_callback(
                        'document_processor',
                        'working',
                        0.8 + (p * 0.2),
                        m
                    )
                )
                
                if success:
                    status_callback(
                        'document_processor',
                        'completed',
                        1.0,
                        "Documents processed successfully"
                    )
                    return True

            status_callback(
                'document_processor',
                'error',
                0,
                "Document processing failed"
            )
            return False
            
        except Exception as e:
            status_callback(
                'document_processor',
                'error',
                0,
                str(e)
            )
            return False

    async def get_relevant_context(self, query: str) -> str:
        """Get relevant context using simple keyword search"""
        try:
            relevant_docs = self.doc_processor.doc_store.search(query, k=3)
            return "\n\n".join(relevant_docs)
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return ""

    async def process_query(
        self,
        query: str,
        status_callback
    ) -> Dict[str, AgentResponse]:
        """Process query through multi-agent system"""
        responses = {}
        context = await self.get_relevant_context(query)
        chat_history = self._format_chat_history()
        
        try:
            # Process through main agent
            status_callback('main_agent', 'working', 0.2, "Analyzing query")
            main_response = await self._get_agent_response(
                'main_agent',
                query,
                context,
                chat_history
            )
            responses['main_agent'] = main_response
            status_callback('main_agent', 'completed', 1.0, "Analysis complete")

            # Process through specialist agents in parallel
            status_callback('diagnosis_agent', 'working', 0.2, "Analyzing symptoms")
            status_callback('treatment_agent', 'working', 0.2, "Evaluating treatments")
            status_callback('research_agent', 'working', 0.2, "Reviewing research")

            specialist_tasks = [
                self._get_agent_response('diagnosis_agent', query, context, chat_history),
                self._get_agent_response('treatment_agent', query, context, chat_history),
                self._get_agent_response('research_agent', query, context, chat_history)
            ]

            specialist_responses = await asyncio.gather(*specialist_tasks)
            
            # Update responses and status for each specialist agent
            for agent_name, response in zip(
                ['diagnosis_agent', 'treatment_agent', 'research_agent'],
                specialist_responses
            ):
                responses[agent_name] = response
                status_callback(
                    agent_name,
                    'completed',
                    1.0,
                    f"{agent_name.split('_')[0].title()} analysis complete"
                )

            # Synthesize final response
            status_callback('synthesis_agent', 'working', 0.5, "Synthesizing insights")
            final_response = await self._synthesize_responses(
                query,
                context,
                chat_history,
                responses
            )
            responses['synthesis_agent'] = final_response
            status_callback(
                'synthesis_agent',
                'completed',
                1.0,
                "Response synthesis complete"
            )

            # Update chat history
            self.chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=final_response.content)
            ])

            return responses

        except Exception as e:
            # Update status for all agents to error state
            for agent in self.agents.keys():
                status_callback(agent, 'error', 0, str(e))
            raise Exception(f"Query processing error: {str(e)}")

    async def _get_agent_response(
        self,
        agent_name: str,
        query: str,
        context: str,
        chat_history: str
    ) -> AgentResponse:
        """Get response from specific agent with metadata"""
        start_time = time.time()
        
        try:
            response = await self.agents[agent_name].ainvoke({
                "input": query,
                "context": context,
                "query": query,
                "chat_history": chat_history
            })
            
            processing_time = time.time() - start_time
            
            metadata = {
                "processing_time": processing_time,
                "context_length": len(context),
                "query_length": len(query)
            }
            
            return AgentResponse(
                agent_name=agent_name,
                content=response,
                confidence=0.85,  # You could implement confidence scoring
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise Exception(f"Agent {agent_name} error: {str(e)}")

    async def _synthesize_responses(
        self,
        query: str,
        context: str,
        chat_history: str,
        responses: Dict[str, AgentResponse]
    ) -> AgentResponse:
        """Synthesize final response from all agent responses"""
        try:
            # Format agent responses for synthesis
            formatted_responses = "\n\n".join([
                f"{name.upper()}:\n{response.content}"
                for name, response in responses.items()
                if name != 'synthesis_agent'
            ])

            start_time = time.time()
            
            synthesis_response = await self.agents['synthesis_agent'].ainvoke({
                "input": query,
                "context": context,
                "query": query,
                "chat_history": chat_history,
                "agent_responses": formatted_responses
            })
            
            processing_time = time.time() - start_time
            
            metadata = {
                "processing_time": processing_time,
                "source_responses": len(responses),
                "context_used": bool(context)
            }
            
            return AgentResponse(
                agent_name="synthesis_agent",
                content=synthesis_response,
                confidence=0.9,
                metadata=metadata,
                processing_time=processing_time
            )

        except Exception as e:
            raise Exception(f"Synthesis error: {str(e)}")


def setup_streamlit_ui():
    """Setup enhanced Streamlit UI with dark sidebar"""
    st.set_page_config(
        page_title="Healthcare AI Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #dee2e6;
            background-color: black;
        }
        .chat-message.user {
            border-left: 4px solid #007bff;
        }
        .chat-message.assistant {
            border-left: 4px solid #28a745;
        }
        .agent-card {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #dee2e6;
            background-color: black;
        }
        .metadata-section {
            font-size: 0.8rem;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
            margin-top: 1rem;
            padding-top: 0.5rem;
        }
        .file-uploader {
            border: 2px dashed #4A4A4A;
            border-radius: 0.5rem;
            padding: 1rem;
            text-align: center;
            background-color: #1E1E1E;
            margin-bottom: 1rem;
            color: #FFFFFF;
        }
        .file-uploader:hover {
            border-color: #007bff;
            background-color: #2D2D2D;
        }
        [data-testid="stSidebar"] {
            background-color: #121212;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: #121212;
        }
        .sidebar .sidebar-content {
            background-color: #121212;
        }
        .stButton>button {
            background-color: #007bff;
            color: black;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.3rem;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .uploaded-file {
            background-color: #1E1E1E;
            padding: 0.8rem;
            border-radius: 0.5rem;
            margin-bottom: 0.8rem;
            border: 1px solid #4A4A4A;
            color: #FFFFFF;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    """Main application with dark sidebar and enhanced UI"""
    setup_streamlit_ui()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = HealthcareAgent()
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = AgentStatus()
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    
    # Sidebar content
    with st.sidebar:
        # Document Processing Section First
        st.markdown('<h3 style="color: #FFFFFF;">📋 Document Processing</h3>', unsafe_allow_html=True)
        
        # Clean single file upload interface
        uploaded_files = st.file_uploader(
            "Upload PDF or Image files",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="document_uploader"
        )
        
        if uploaded_files:
            st.markdown('<h4 style="color: #FFFFFF;">📎 Selected Files</h4>', unsafe_allow_html=True)
            for file in uploaded_files:
                st.markdown(f"""
                    <div class="uploaded-file">
                        <div style="color: #FFFFFF;">📄 {file.name}</div>
                        <div style="color: #CCCCCC; font-size: 0.8rem; margin-top: 0.5rem;">
                            Type: {file.type}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            if st.button("🔄 Process Documents", key="process_docs"):
                with st.spinner("Processing documents..."):
                    async def process_docs():
                        await st.session_state.agent.process_documents(
                            uploaded_files,
                            st.session_state.agent_status.update_status
                        )
                        st.session_state.documents_processed = True
                    
                    asyncio.run(process_docs())
                    
        st.markdown('<h3 style="color: #FFFFFF;">🤖 Agent Status</h3>', unsafe_allow_html=True)
        # Initialize agent status sidebar after file upload section
        st.session_state.agent_status.initialize_sidebar_placeholder()
    
    # Main content area
    st.title("🏥 Healthcare AI Assistant")
    st.markdown("""
        Your intelligent medical assistant for document analysis and healthcare queries.
        Upload documents in the sidebar and ask questions below.
    """)
    
    # Chat interface
    st.markdown("### 💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                # Display synthesized response
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        {message['content']['synthesis_agent'].content}
                    </div>
                """, unsafe_allow_html=True)
                
                # Show detailed agent responses in expander
                with st.expander("🔍 Detailed Agent Responses", expanded=False):
                    for agent_name, response in message['content'].items():
                        if agent_name != 'synthesis_agent':
                            st.markdown(f"""
                                <div class="agent-card">
                                    <strong>{agent_name.replace('_', ' ').title()}</strong>
                                    <div style="margin: 0.5rem 0;">
                                        {response.content}
                                    </div>
                                    <div class="metadata-section">
                                        Processing time: {response.processing_time:.2f}s
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        {message['content']}
                    </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me about your health concerns..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            try:
                async def process_query():
                    return await st.session_state.agent.process_query(
                        prompt,
                        st.session_state.agent_status.update_status
                    )
                
                responses = asyncio.run(process_query())
                
                if responses:
                    response_placeholder.markdown(f"""
                        <div class="chat-message assistant">
                            {responses['synthesis_agent'].content}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": responses
                    })
                
            except Exception as e:
                response_placeholder.error(f"An error occurred: {str(e)}")
                
if __name__ == "__main__":
    main()
