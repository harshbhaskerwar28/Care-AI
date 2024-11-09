import streamlit as st
from typing import Dict, Callable, Optional
from dataclasses import dataclass

@dataclass
class AgentStatusConfig:
    """Configuration for agent status appearance and behavior"""
    colors: Dict[str, str] = None
    icons: Dict[str, str] = None
    
    def __post_init__(self):
        self.colors = self.colors or {
            'idle': '#6c757d',
            'working': '#007bff',
            'completed': '#28a745',
            'error': '#dc3545'
        }
        self.icons = self.icons or {
            'idle': '⚪',
            'working': '🔄',
            'completed': '✅',
            'error': '❌'
        }

class AgentStatus:
    """Enhanced agent status management with detailed sidebar display"""
    def __init__(self, config: Optional[AgentStatusConfig] = None):
        self.sidebar_placeholder = None
        self.config = config or AgentStatusConfig()
        self.agents = {
            'document_processor': {'status': 'idle', 'progress': 0, 'message': ''},
            'main_agent': {'status': 'idle', 'progress': 0, 'message': ''},
            'diagnosis_agent': {'status': 'idle', 'progress': 0, 'message': ''},
            'treatment_agent': {'status': 'idle', 'progress': 0, 'message': ''},
            'research_agent': {'status': 'idle', 'progress': 0, 'message': ''},
            'synthesis_agent': {'status': 'idle', 'progress': 0, 'message': ''}
        }
        
    def initialize_sidebar_placeholder(self):
        """Initialize the sidebar placeholder with custom styling"""
        with st.sidebar:
            st.markdown("""
                <style>
                .agent-status-container {
                    background-color: #1E1E1E;
                    border-radius: 0.5rem;
                    margin-bottom: 1rem;
                    padding: 0.5rem;
                }
                .status-header {
                    color: #FFFFFF;
                    font-size: 1.2rem;
                    margin-bottom: 1rem;
                    padding-bottom: 0.5rem;
                    border-bottom: 1px solid #333;
                }
                </style>
            """, unsafe_allow_html=True)
            self.sidebar_placeholder = st.empty()
    
    def update_status(self, agent_name: str, status: str, progress: float = 0, message: str = ""):
        """Update agent status with validation and refresh sidebar display"""
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        if status not in self.config.colors:
            raise ValueError(f"Invalid status: {status}")
        if not 0 <= progress <= 1:
            raise ValueError("Progress must be between 0 and 1")
            
        self.agents[agent_name] = {
            'status': status,
            'progress': progress,
            'message': message
        }
        self._render_status()

    def _render_status(self):
        """Render all agent statuses in sidebar with enhanced styling"""
        if self.sidebar_placeholder is None:
            self.initialize_sidebar_placeholder()
            
        with self.sidebar_placeholder.container():
            st.markdown('<div class="status-header">🤖 Agent Status</div>', unsafe_allow_html=True)
            for agent_name, status in self.agents.items():
                self._render_agent_card(agent_name, status)

    def _render_agent_card(self, agent_name: str, status: dict):
        """Render individual agent status card with enhanced styling"""
        color = self.config.colors[status['status']]
        icon = self.config.icons[status['status']]
        
        st.markdown(f"""
            <div style="
                background-color: #1E1E1E;
                padding: 0.8rem;
                border-radius: 0.5rem;
                margin-bottom: 0.8rem;
                border: 1px solid {color};
                animation: fadeIn 0.3s ease-in;
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    color: {color};
                    font-weight: bold;
                    margin-bottom: 0.5rem;
                ">
                    <span>{agent_name.replace('_', ' ').title()}</span>
                    <span>{icon}</span>
                </div>
                <div style="
                    color: #CCCCCC;
                    font-size: 0.8rem;
                    margin: 0.3rem 0;
                ">
                    {status['message'] or status['status'].title()}
                </div>
                <div style="
                    height: 4px;
                    background-color: rgba(255,255,255,0.1);
                    border-radius: 2px;
                    margin-top: 0.5rem;
                ">
                    <div style="
                        width: {status['progress'] * 100}%;
                        height: 100%;
                        background-color: {color};
                        border-radius: 2px;
                        transition: width 0.3s ease;
                    "></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    def reset_all(self):
        """Reset all agents to idle state"""
        for agent_name in self.agents:
            self.agents[agent_name] = {
                'status': 'idle',
                'progress': 0,
                'message': ''
            }
        self._render_status()

    def get_agent_status(self, agent_name: str) -> dict:
        """Get current status of specific agent"""
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        return self.agents[agent_name].copy()

# Helper functions
def display_agent_status():
    """Initialize agent status in session state"""
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = AgentStatus()
    
    if st.session_state.agent_status.sidebar_placeholder is None:
        st.session_state.agent_status.initialize_sidebar_placeholder()

def update_agent_status(agent_name: str, status: str, progress: float = 0, message: str = ""):
    """Update agent status with session state handling"""
    if 'agent_status' in st.session_state:
        st.session_state.agent_status.update_status(agent_name, status, progress, message)

# Example usage in a Streamlit app
def main():
    st.set_page_config(page_title="Agent Status Demo", layout="wide")
    
    # Initialize agent status
    display_agent_status()
    
    # Demo controls
    st.sidebar.title("Demo Controls")
    agent_name = st.sidebar.selectbox("Select Agent", list(st.session_state.agent_status.agents.keys()))
    status = st.sidebar.selectbox("Status", ['idle', 'working', 'completed', 'error'])
    progress = st.sidebar.slider("Progress", 0.0, 1.0, 0.0)
    message = st.sidebar.text_input("Status Message")
    
    if st.sidebar.button("Update Status"):
        update_agent_status(agent_name, status, progress, message)
    
    if st.sidebar.button("Reset All"):
        st.session_state.agent_status.reset_all()

if __name__ == "__main__":
    main()
