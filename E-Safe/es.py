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
import pathlib

# Configure NLTK data path to a writable directory
nltk_data_dir = os.path.join(pathlib.Path.home(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download NLTK data with error handling
def download_nltk_data():
    try:
        for package in ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']:
            try:
                nltk.download(package, download_dir=nltk_data_dir, quiet=True)
            except Exception as e:
                logging.warning(f"Failed to download NLTK package {package}: {e}")
                continue
    except Exception as e:
        logging.error(f"NLTK data download failed: {e}")

# Call NLTK download function
download_nltk_data()

# Load environment variables
load_dotenv()

# Tokens and IDs with proper error handling
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logging.warning("TELEGRAM_BOT_TOKEN not found in environment variables")
    TELEGRAM_BOT_TOKEN = "dummy_token"

ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")
if not ADMIN_CHAT_ID:
    logging.warning("ADMIN_CHAT_ID not found in environment variables")
    ADMIN_CHAT_ID = "dummy_chat_id"

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("emergency_app.log")
    ]
)
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

                if lat and lon:  # Check if coordinates are valid
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
                        location = geolocator.reverse(f"{lat}, {lon}", timeout=10)
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
                location = geolocator.geocode(emergency_details['text_address'], timeout=10)
                if location:
                    maps_link = f"https://www.google.com/maps?q={location.latitude},{location.longitude}"
                    alert_message += f"🗺️ Address Google Maps: {maps_link}\n"
            except Exception as geo_error:
                logger.error(f"Address geocoding error: {geo_error}")

        # Send text message with error handling
        try:
            message_data = {
                "chat_id": ADMIN_CHAT_ID,
                "text": alert_message,
                "parse_mode": "HTML"
            }
            response = requests.post(f"{base_url}/sendMessage", json=message_data, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send message: {e}")
            return False

        # Send photos if any
        if uploaded_files:
            for file in uploaded_files:
                try:
                    files = {"photo": file.getvalue()}
                    photo_data = {
                        "chat_id": ADMIN_CHAT_ID,
                        "caption": "Emergency situation photo"
                    }
                    response = requests.post(f"{base_url}/sendPhoto", data=photo_data, files=files, timeout=10)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to send photo: {e}")
                    continue

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

@st.cache_resource
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
    try:
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
                try:
                    map_center = [20.5937, 78.9629]  # Example center location
                    m = folium.Map(location=map_center, zoom_start=5)
                    map_data = st_folium(m, width=700, height=500)

                    if map_data["last_clicked"]:
                        latitude, longitude = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
                        st.session_state.current_location = {"latitude": latitude, "longitude": longitude}
                        st.session_state.step = 'photos'
                        st.success(f"Location captured: {latitude}, {longitude}")
                        st.rerun()
                except Exception as e:
                    logger.error(f"Map error: {e}")
                    st.error("Failed to load map. Please try entering your address instead.")
                    st.session_state.step = 'text_address'
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

            # Display location information if available
            if st.session_state.current_location:
                try:
                    location_data = st.session_state.current_location
                    if isinstance(location_data, dict):
                        lat, lon = location_data.get('latitude'), location_data.get('longitude')
                    else:
                        lat, lon = map(float, location_data.split(','))
                    
                    custom_card(
                        "📍 Your Location",
                        f"Coordinates: {lat}, {lon}",
                        "#9C27B0"
                    )
                    
                    # Show map with location
                    m = folium.Map(location=[lat, lon], zoom_start=15)
                    folium.Marker(
                        [lat, lon],
                        popup="Emergency Location",
                        icon=folium.Icon(color='red', icon='info-sign')
                    ).add_to(m)
                    st_folium(m, width=700, height=300)
                except Exception as e:
                    logger.error(f"Error displaying location map: {e}")

            elif st.session_state.text_address:
                custom_card(
                    "📍 Your Address",
                    st.session_state.text_address,
                    "#9C27B0"
                )

            # Status updates section
            status_updates = {
                "Dispatch": "✅ Emergency services notified",
                "En Route": f"🚑 Estimated arrival in {st.session_state.estimated_time} minutes",
                "Preparation": "⏳ Emergency team preparing for arrival"
            }

            custom_card(
                "📊 Status Updates",
                "\n".join([f"{key}: {value}" for key, value in status_updates.items()]),
                "#1E88E5"
            )

            # Emergency type specific instructions
            emergency_type = st.session_state.emergency_type
            if emergency_type:
                specific_instructions = {
                    "Medical Emergency": """
                    • List any medications you're currently taking
                    • Note any allergies
                    • Gather recent medical records if available
                    """,
                    "Accident": """
                    • Don't move if seriously injured
                    • Take photos of the scene if safe to do so
                    • Gather witness information if applicable
                    """,
                    "Heart/Chest Pain": """
                    • Sit or lie down and try to stay calm
                    • Loosen any tight clothing
                    • Take note of any other symptoms
                    • If prescribed, take nitroglycerin as directed
                    """,
                    "Pregnancy": """
                    • Time contractions if applicable
                    • Prepare hospital bag if not done
                    • Have medical records ready
                    • Stay hydrated
                    """
                }
                
                if emergency_type in specific_instructions:
                    custom_card(
                        f"📋 Specific Instructions for {emergency_type}",
                        specific_instructions[emergency_type],
                        "#FF9800"
                    )

            # Cancelation warning
            custom_card(
                "⚠️ Important Notice",
                "Only cancel emergency services if you are absolutely sure you no longer need assistance.",
                "#FF4B4B"
            )

            # Add buttons for additional actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📞 Call Emergency Services", use_container_width=True):
                    st.markdown("Dialing emergency services...")
                    # In a real application, this would trigger a phone call

            with col2:
                if st.button("⚠️ Cancel Emergency", use_container_width=True, type="secondary"):
                    st.warning("Are you sure you want to cancel emergency services?")
                    col3, col4 = st.columns(2)
                    with col3:
                        if st.button("Yes, Cancel", use_container_width=True):
                            # Reset session state
                            for key in st.session_state.keys():
                                del st.session_state[key]
                            st.success("Emergency services have been canceled.")
                            st.rerun()
                    with col4:
                        if st.button("No, Keep Active", use_container_width=True):
                            st.info("Emergency services will continue to respond.")

            # Reset button (bottom of page)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Start New Emergency Request", use_container_width=True):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please try again or contact support.")
        if st.button("Restart Application"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
            
