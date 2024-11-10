import streamlit as st
import os
import logging
from dotenv import load_dotenv
import nltk
import re
from geopy.geocoders import Nominatim
from PIL import Image
import io
import requests
from datetime import datetime
import urllib.parse
import folium
from streamlit_folium import st_folium

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# Load environment variables
load_dotenv()

# Tokens and IDs
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

# Configure logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def send_emergency_alert_to_admin(emergency_details, uploaded_files):
    """Send emergency details and images to admin chat"""
    try:
        base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
        
        alert_message = (
            "🚨 NEW EMERGENCY ALERT 🚨\n\n"
            f"Type: {emergency_details['type']}\n"
            f"Time: {emergency_details['time']}\n\n"
        )

        # Handle location information
        if emergency_details.get('current_location'):
            try:
                # Parse location string to get coordinates
                if isinstance(emergency_details['current_location'], str):
                    lat, lon = map(float, emergency_details['current_location'].split(','))
                else:
                    lat = emergency_details['current_location'].get('latitude')
                    lon = emergency_details['current_location'].get('longitude')

                # Create Google Maps link
                maps_link = f"https://www.google.com/maps?q={lat},{lon}"
                
                # Add location information to message
                alert_message += (
                    f"📍 Location Coordinates: {lat}, {lon}\n"
                    f"🗺️ Google Maps: {maps_link}\n"
                )

                # Try to get address from coordinates using Nominatim
                try:
                    geolocator = Nominatim(user_agent="emergency_app")
                    location = geolocator.reverse(f"{lat}, {lon}")
                    if location and location.address:
                        alert_message += f"📌 Reverse Geocoded Address: {location.address}\n"
                except Exception as geo_error:
                    logger.error(f"Geocoding error: {geo_error}")
                    
            except Exception as loc_error:
                logger.error(f"Location parsing error: {loc_error}")
                alert_message += f"📍 Location (raw): {emergency_details['current_location']}\n"

        if emergency_details.get('text_address'):
            alert_message += f"🏠 Provided Address: {emergency_details['text_address']}\n"
            # Try to get coordinates for the text address
            try:
                geolocator = Nominatim(user_agent="emergency_app")
                location = geolocator.geocode(emergency_details['text_address'])
                if location:
                    maps_link = f"https://www.google.com/maps?q={location.latitude},{location.longitude}"
                    alert_message += f"🗺️ Address Google Maps: {maps_link}\n"
            except Exception as geo_error:
                logger.error(f"Address geocoding error: {geo_error}")

        # Send text message
        message_data = {
            "chat_id": ADMIN_CHAT_ID,
            "text": alert_message,
            "parse_mode": "HTML"
        }
        requests.post(f"{base_url}/sendMessage", json=message_data)

        # Send photos if any
        if uploaded_files:
            for file in uploaded_files:
                files = {"photo": file.getvalue()}
                photo_data = {
                    "chat_id": ADMIN_CHAT_ID,
                    "caption": "Emergency situation photo"
                }
                requests.post(f"{base_url}/sendPhoto", data=photo_data, files=files)

        return True
    except Exception as e:
        logger.error(f"Failed to send emergency alert: {e}")
        return False

def custom_card(title, content=None, color="#FF4B4B"):
    st.markdown(
        f"""
        <div style="
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            background-color: #1E1E1E;
            border-left: 5px solid {color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);">
            <h3 style="color: {color}; margin-top: 0;">{title}</h3>
            {f'<p style="color: #E0E0E0; margin-bottom: 0;">{content}</p>' if content else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

def initialize_session_state():
    """Initialize session state variables"""
    if 'step' not in st.session_state:
        st.session_state.step = 'platform_choice'
    if 'platform' not in st.session_state:
        st.session_state.platform = None
    if 'emergency_type' not in st.session_state:
        st.session_state.emergency_type = None
    if 'current_location' not in st.session_state:
        st.session_state.current_location = None
    if 'text_address' not in st.session_state:
        st.session_state.text_address = None
    if 'location_choice' not in st.session_state:
        st.session_state.location_choice = None
    if 'photos' not in st.session_state:
        st.session_state.photos = []
    if 'alert_sent' not in st.session_state:
        st.session_state.alert_sent = False
    if 'emergency_status' not in st.session_state:
        st.session_state.emergency_status = None

def get_estimated_time():
    """Return a random estimated arrival time between 5-15 minutes"""
    from random import randint
    return randint(5, 15)


def main():
    # Set dark theme
    st.set_page_config(
        page_title="Emergency Assistance",
        page_icon="🚑",
        layout="centered",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )

    # Initialize session state
    initialize_session_state()

    # Custom CSS for dark theme
    st.markdown("""
        <style>
        /* Dark theme styles */
        body {
            background-color: #121212;
            color: #E0E0E0;
        }
        .main {
            padding: 2rem;
            max-width: 900px;
            margin: 0 auto;
            background-color: #121212;
        }
        .stButton button {
            width: 100%;
            border-radius: 20px;
            height: 3em;
            font-weight: 600;
            background-color: #2C2C2C;
            color: #E0E0E0;
            border: 1px solid #404040;
        }
        .stButton button:hover {
            background-color: #404040;
            border-color: #505050;
        }
        .emergency-title {
            color: #FF4B4B;
            text-align: center;
            margin-bottom: 2em;
        }
        .stTextInput input, .stTextArea textarea {
            background-color: #2C2C2C;
            color: #E0E0E0;
            border: 1px solid #404040;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #505050;
            box-shadow: 0 0 0 1px #505050;
        }
        .uploadedFile {
            background-color: #2C2C2C;
            color: #E0E0E0;
            border: 1px solid #404040;
        }
        .css-1d391kg {
            background-color: #1E1E1E;
        }
        .folium-map {
            border: 2px solid #404040;
            border-radius: 10px;
        }
        /* Override Streamlit's default white background */
        .stApp {
            background-color: #121212;
        }
        </style>
    """, unsafe_allow_html=True)

@@ -238,9 +282,13 @@ def main():

        elif st.session_state.step == 'current_location':
            custom_card("Select Your Location on the Map", color="#4CAF50")
            # Create a dark-themed map
            map_center = [20.5937, 78.9629]  # Example center location
            m = folium.Map(
                location=map_center,
                zoom_start=5,
                tiles="cartodbdark_matter"  # Dark theme tiles
            )
            map_data = st_folium(m, width=700, height=500)

            if map_data["last_clicked"]:
