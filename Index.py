import streamlit as st
import base64
from PIL import Image
import streamlit.components.v1 as components

# Configure the page settings
st.set_page_config(
    page_title="Multi-Project Dashboard",
    page_icon="🏠",
    layout="wide",
)

# Apply dark theme styling
dark_theme_css = """
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .project-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    .project-card:hover {
        transform: translateY(-5px);
    }
    .card-title {
        color: #FFFFFF;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .card-description {
        color: #C2C2C2;
        font-size: 14px;
    }
</style>
"""
st.markdown(dark_theme_css, unsafe_allow_html=True)

def load_and_resize_logo(logo_path, width=150):
    """Load and resize the logo image"""
    try:
        image = Image.open(logo_path)
        # Maintain aspect ratio while resizing
        aspect_ratio = image.size[1] / image.size[0]
        height = int(width * aspect_ratio)
        image = image.resize((width, height))
        return image
    except Exception as e:
        st.error(f"Error loading logo: {e}")
        return None

def create_project_card(title, description, key):
    """Create a clickable project card with custom styling"""
    card_html = f"""
    <div class="project-card" onclick="handle_click_{key}()">
        <div class="card-title">{title}</div>
        <div class="card-description">{description}</div>
    </div>
    <script>
        function handle_click_{key}() {{
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                value: '{key}'
            }}, '*');
        }}
    </script>
    """
    return card_html

def main():
    # Header section with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        # Replace 'path_to_your_logo.png' with your actual logo path
        logo = load_and_resize_logo('path_to_your_logo.png')
        if logo:
            st.image(logo)

    # Title in center column
    with col2:
        st.title("Project Dashboard")

    # Create two rows with two cards each
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # Project 1
    with row1_col1:
        components.html(
            create_project_card(
                "Project 1",
                "Description of your first project",
                "project1"
            ),
            height=200
        )
        if st.session_state.get('selected_project') == 'project1':
            st.write("Project 1 Content")
            # ===== ADD YOUR PROJECT 1 CODE HERE =====
            # def project1_main():
            #     Your code here
            # project1_main()

    # Project 2
    with row1_col2:
        components.html(
            create_project_card(
                "Project 2",
                "Description of your second project",
                "project2"
            ),
            height=200
        )
        if st.session_state.get('selected_project') == 'project2':
            st.write("Project 2 Content")
            # ===== ADD YOUR PROJECT 2 CODE HERE =====
            # def project2_main():
            #     Your code here
            # project2_main()

    # Project 3
    with row2_col1:
        components.html(
            create_project_card(
                "Project 3",
                "Description of your third project",
                "project3"
            ),
            height=200
        )
        if st.session_state.get('selected_project') == 'project3':
            st.write("Project 3 Content")
            # ===== ADD YOUR PROJECT 3 CODE HERE =====
            # def project3_main():
            #     Your code here
            # project3_main()

    # Project 4
    with row2_col2:
        components.html(
            create_project_card(
                "Project 4",
                "Description of your fourth project",
                "project4"
            ),
            height=200
        )
        if st.session_state.get('selected_project') == 'project4':
            st.write("Project 4 Content")
            # ===== ADD YOUR PROJECT 4 CODE HERE =====
            # def project4_main():
            #     Your code here
            # project4_main()

    # Handle card clicks
    for key in ['project1', 'project2', 'project3', 'project4']:
        if key not in st.session_state:
            st.session_state[key] = False

    # Initialize selected_project in session state if not already present
    if 'selected_project' not in st.session_state:
        st.session_state['selected_project'] = None

if __name__ == "__main__":
    main()
