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
            background-color: white;
            border-left: 5px solid {color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="color: {color}; margin-top: 0;">{title}</h3>
            {f'<p style="color: #000000; margin-bottom: 0;">{content}</p>' if content else ''}
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
    st.set_page_config(
        page_title="Emergency Assistance",
        page_icon="🚑",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state
    initialize_session_state()

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
            max-width: 900px;
            margin: 0 auto;
        }
        .stButton button {
            width: 100%;
            border-radius: 20px;
            height: 3em;
            font-weight: 600;
        }
        .emergency-title {
            color: #FF4B4B;
            text-align: center;
            margin-bottom: 2em;
        }
        </style>
    """, unsafe_allow_html=True)

    if not st.session_state.alert_sent:
        st.markdown('<h1 class="emergency-title">🚑 Emergency Assistance</h1>', unsafe_allow_html=True)

        if st.session_state.step == 'platform_choice':
            custom_card("Choose how you'd like to continue", color="#1E88E5")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Continue Here", use_container_width=True):
                    st.session_state.platform = "streamlit"
                    st.session_state.step = 'emergency_type'
                    st.rerun()
            with col2:
                if st.button("Open in Telegram", use_container_width=True):
                    bot_username = "EmergencyEagleBot"
                    telegram_url = f"https://t.me/{bot_username}"
                    st.markdown(f"[Open Telegram Bot]({telegram_url})")
                    st.stop()

        elif st.session_state.step == 'emergency_type':
            custom_card("Select Emergency Type", color="#FF4B4B")
            emergency_options = {
                "Medical Emergency": "🏥",
                "Accident": "🚗",
                "Heart/Chest Pain": "❤️",
                "Pregnancy": "👶"
            }
            
            cols = st.columns(2)
            for i, (option, emoji) in enumerate(emergency_options.items()):
                with cols[i % 2]:
                    if st.button(f"{emoji} {option}", use_container_width=True):
                        st.session_state.emergency_type = option
                        st.session_state.step = 'location_choice'
                        st.rerun()

        elif st.session_state.step == 'location_choice':
            custom_card("Share Your Location", color="#4CAF50")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📍 Share Location", use_container_width=True):
                    st.session_state.location_choice = "location"
                    st.session_state.step = 'current_location'
                    st.rerun()
                
            with col2:
                if st.button("✍️ Enter Address", use_container_width=True):
                    st.session_state.location_choice = "address"
                    st.session_state.step = 'text_address'
                    st.rerun()

        elif st.session_state.step == 'current_location':
            custom_card("Select Your Location on the Map", color="#4CAF50")
            # Display the map with a folium map to select location
            map_center = [20.5937, 78.9629]  # Example center location
            m = folium.Map(location=map_center, zoom_start=5)
            map_data = st_folium(m, width=700, height=500)

            if map_data["last_clicked"]:
                latitude, longitude = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
                st.session_state.current_location = {"latitude": latitude, "longitude": longitude}
                st.session_state.step = 'photos'
                st.success(f"Location captured: {latitude}, {longitude}")
                st.rerun()

        elif st.session_state.step == 'text_address':
            custom_card("Enter Your Address", color="#4CAF50")
            text_address = st.text_area("Complete Address")
            if st.button("Continue", use_container_width=True):
                if text_address:
                    st.session_state.text_address = text_address
                    st.session_state.step = 'photos'
                    st.rerun()
                else:
                    st.error("Please enter your address")

        elif st.session_state.step == 'photos':
            custom_card("Upload Photos (Optional)", color="#9C27B0")
            uploaded_files = st.file_uploader(
                "Upload photos of the emergency situation",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True
            )
            if st.button("Send Emergency Alert", use_container_width=True):
                st.session_state.photos = uploaded_files
                st.session_state.step = 'summary'
                st.rerun()

        elif st.session_state.step == 'summary':
            emergency_details = {
                'type': st.session_state.emergency_type,
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'current_location': st.session_state.current_location,
                'text_address': st.session_state.text_address
            }

            with st.spinner("Dispatching Emergency Services..."):
                if send_emergency_alert_to_admin(emergency_details, st.session_state.photos):
                    st.session_state.alert_sent = True
                    st.session_state.emergency_status = "en_route"
                    estimated_time = get_estimated_time()
                    st.session_state.estimated_time = estimated_time
                    st.rerun()
                else:
                    st.error("Failed to send alert. Please try again.")

    else:
        # Emergency services dispatched view
        st.markdown('<h1 class="emergency-title">Emergency Services En Route</h1>', unsafe_allow_html=True)
        
        custom_card(
            "🚑 Help is on the way!",
            f"Estimated arrival time: {st.session_state.estimated_time} minutes",
            "#4CAF50"
        )

        custom_card(
            "📝 Important Instructions",
            """
            • Stay calm and remain in your current location
            • Keep your phone nearby
            • Gather any relevant medical documents
            • Clear the path for emergency responders
            • If possible, have someone wait outside to guide the team
            """,
            "#1E88E5"
        )

        custom_card(
            "🆘 Emergency Contact",
            "If your condition worsens or you need immediate assistance, call 911",
            "#FF4B4B"
        )

        # Reset button (bottom of page)
        if st.button("Start New Emergency Request", use_container_width=True):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
