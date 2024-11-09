import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import base64

# Configure the page settings
st.set_page_config(
    page_title="Project Hub",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS for dark theme and styling
def apply_custom_style():
    st.markdown("""
        <style>
        /* Dark theme background */
        .stApp {
            background-color: #1E1E1E;
            color: white;
        }
        
        /* Custom button styling */
        .custom-button {
            background-color: #2E2E2E;
            color: white;
            padding: 20px 40px;
            border-radius: 10px;
            border: 1px solid #4E4E4E;
            cursor: pointer;
            margin: 10px;
            width: 250px;
            font-size: 18px;
            transition: all 0.3s;
        }
        
        .custom-button:hover {
            background-color: #4E4E4E;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        /* Container styling */
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        
        /* Logo styling */
        .logo-container {
            position: absolute;
            top: 20px;
            left: 20px;
        }
        
        /* Title styling */
        .title {
            color: white;
            font-size: 48px;
            text-align: center;
            margin-bottom: 50px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    apply_custom_style()
    
    # Add logo (replace 'logo.png' with your logo path)
    # st.image("logo.png", width=100)  # Uncomment and add your logo
    
    # Main title
    st.markdown('<p class="title">Project Hub</p>', unsafe_allow_html=True)
    
    # Center container for buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Project 1 Button
        if st.button("Project 1", key="btn1", help="Click to open Project 1"):
            st.session_state.current_project = "project1"
            
        # Project 2 Button
        if st.button("Project 2", key="btn2", help="Click to open Project 2"):
            st.session_state.current_project = "project2"
            
        # Project 3 Button
        if st.button("Project 3", key="btn3", help="Click to open Project 3"):
            st.session_state.current_project = "project4"
            
        # Project 4 Button
        if st.button("Project 4", key="btn4", help="Click to open Project 4"):
            st.session_state.current_project = "project4"

    # Initialize session state for project selection if not exists
    if 'current_project' not in st.session_state:
        st.session_state.current_project = None

    # Display selected project
    if st.session_state.current_project == "project1":
        # ===== PROJECT 1 CODE =====
        # Add your Project 1 code here
        st.markdown("### Project 1")
        # def project1_main():
        #     Your project 1 code here
        # project1_main()
        
    elif st.session_state.current_project == "project2":
        # ===== PROJECT 2 CODE =====
        # Add your Project 2 code here
        st.markdown("### Project 2")
        # def project2_main():
        #     Your project 2 code here
        # project2_main()
        
    elif st.session_state.current_project == "project3":
        # ===== PROJECT 3 CODE =====
        # Add your Project 3 code here
        st.markdown("### Project 3")
        # def project3_main():
        #     Your project 3 code here
        # project3_main()
        
    elif st.session_state.current_project == "project4":
        # ===== PROJECT 4 CODE =====
        # Add your Project 4 code here
        st.markdown("### Project 4")
        # def project4_main():
        #     Your project 4 code here
        # project4_main()

if __name__ == "__main__":
    main()
