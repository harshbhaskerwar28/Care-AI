import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json

# Initialize session state if not exists
if 'current_prediction_type' not in st.session_state:
    st.session_state.current_prediction_type = None

# Color scheme
COLOR_SCHEME = {
    'primary': '#4F46E5',
    'secondary': '#818CF8',
    'success': '#34D399',
    'warning': '#FBBF24',
    'danger': '#EF4444',
    'background': '#F3F4F6',
    'text': '#000000'
}

# Load configuration
config = {
    "prediction_types": {
        "diabetes": {
            "name": "Diabetes Risk Assessment",
            "icon": "🩺",
            "required_metrics": ["glucose_fasting", "glucose_post_meal", "bmi"],
            "description": "Evaluates your risk of diabetes based on blood glucose levels and body composition",
            "recommended_frequency": "Every 6 months"
        },
        "heart": {
            "name": "Heart Health Screening",
            "icon": "❤️",
            "required_metrics": ["heart_rate", "bp_systolic", "bp_diastolic", "bmi"],
            "description": "Assesses your cardiovascular health based on vital signs and body composition",
            "recommended_frequency": "Every 3 months"
        },
        "obesity": {
            "name": "Weight Status Analysis",
            "icon": "⚖️",
            "required_metrics": ["bmi"],
            "description": "Evaluates your weight status and related health risks",
            "recommended_frequency": "Monthly"
        }
    },
    "metrics": {
        "glucose": {
            "fasting": {
                "ranges": {
                    "normal": {"min": 70, "max": 99},
                    "prediabetes": {"min": 100, "max": 125},
                    "diabetes": {"min": 126, "max": 200}
                }
            },
            "post_meal": {
                "ranges": {
                    "normal": {"min": 70, "max": 139},
                    "prediabetes": {"min": 140, "max": 199},
                    "diabetes": {"min": 200, "max": 300}
                }
            }
        },
        "vitals": {
            "heart_rate": {
                "ranges": {
                    "low": {"min": 40, "max": 59},
                    "normal": {"min": 60, "max": 100},
                    "high": {"min": 101, "max": 130}
                }
            },
            "blood_pressure_systolic": {
                "ranges": {
                    "normal": {"min": 90, "max": 120},
                    "elevated": {"min": 121, "max": 129},
                    "high": {"min": 130, "max": 180}
                }
            },
            "blood_pressure_diastolic": {
                "ranges": {
                    "normal": {"min": 60, "max": 80},
                    "high": {"min": 81, "max": 120}
                }
            }
        },
        "bmi": {
            "ranges": {
                "underweight": {"min": 0, "max": 18.4},
                "normal": {"min": 18.5, "max": 24.9},
                "overweight": {"min": 25, "max": 29.9},
                "obese": {"min": 30, "max": 50}
            }
        }
    }
}

def calculate_bmi(weight, height):
    """Calculate BMI from weight (kg) and height (m)"""
    try:
        bmi = weight / (height ** 2)
        return round(bmi, 1)
    except:
        return 0

def get_bmi_category(bmi):
    """Get BMI category based on value"""
    ranges = config["metrics"]["bmi"]["ranges"]
    for category, range_values in ranges.items():
        if range_values["min"] <= bmi <= range_values["max"]:
            return category
    return "undefined"

def create_gauge_chart(value, ranges, title):
    """Create a gauge chart for displaying health metrics"""
    min_val = min(r['min'] for r in ranges.values())
    max_val = max(r['max'] for r in ranges.values())
    
    steps = []
    colors = [COLOR_SCHEME['success'], COLOR_SCHEME['warning'], COLOR_SCHEME['danger']]
    
    for (range_name, range_values), color in zip(ranges.items(), colors):
        steps.append({
            'range': [range_values['min'], range_values['max']], 
            'color': color,
            'name': range_name
        })
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': COLOR_SCHEME['text']}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'steps': steps,
            'threshold': {
                'line': {'color': COLOR_SCHEME['primary'], 'width': 4},
                'thickness': 0.75,
                'value': value
            },
            'bar': {'color': COLOR_SCHEME['primary']}
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLOR_SCHEME['text'], 'family': "Inter"}
    )
    
    return fig

def get_health_insights(metrics, prediction_type):
    """Generate health insights based on metrics"""
    insights = []
    
    if prediction_type == "diabetes":
        # Glucose insights
        if metrics.get('glucose_fasting'):
            if metrics['glucose_fasting'] < 100:
                insights.append("\n • Your fasting glucose is in the normal range.")
            elif metrics['glucose_fasting'] < 126:
                insights.append("\n • Your fasting glucose indicates pre-diabetes risk.")
            else:
                insights.append("\n • Your fasting glucose is elevated, suggesting diabetes risk.")
        
        if metrics.get('glucose_post_meal'):
            if metrics['glucose_post_meal'] < 140:
                insights.append("\n • Your post-meal glucose is normal.")
            elif metrics['glucose_post_meal'] < 200:
                insights.append("\n • Your post-meal glucose suggests pre-diabetes risk.")
            else:
                insights.append("\n • Your post-meal glucose is high, indicating diabetes risk.")
    
    elif prediction_type == "heart":
        # Blood pressure insights
        if metrics.get('bp_systolic') and metrics.get('bp_diastolic'):
            if metrics['bp_systolic'] < 120 and metrics['bp_diastolic'] < 80:
                insights.append("\n • Your blood pressure is in the normal range.")
            elif metrics['bp_systolic'] < 130 and metrics['bp_diastolic'] < 80:
                insights.append("\n • Your blood pressure is slightly elevated.")
            else:
                insights.append("\n • Your blood pressure is high. Consider lifestyle changes.")
        
        # Heart rate insights
        if metrics.get('heart_rate'):
            if 60 <= metrics['heart_rate'] <= 100:
                insights.append("\n • Your resting heart rate is normal.")
            elif metrics['heart_rate'] < 60:
                insights.append("\n • Your heart rate is low. This might be normal for athletes.")
            else:
                insights.append("\n • Your heart rate is elevated.")
    
    # BMI insights
    if metrics.get('bmi'):
        bmi_category = get_bmi_category(metrics['bmi'])
        if bmi_category == "normal":
            insights.append("\n • Your BMI is in the healthy range.")
        elif bmi_category == "underweight":
            insights.append("\n • Your BMI indicates you may be underweight.")
        elif bmi_category == "overweight":
            insights.append("\n • Your BMI indicates you may be overweight.")
        elif bmi_category == "obese":
            insights.append("\n • Your BMI indicates obesity. Consider consulting a healthcare provider.")
    
    # Add general recommendations
    insights.append("\n**General Recommendations:**")
    insights.append("\n • Maintain a balanced diet rich in whole foods")
    insights.append("\n • Engage in regular physical activity (150 minutes per week)")
    insights.append("\n • Get 7-9 hours of quality sleep each night")
    insights.append("\n • Manage stress through relaxation techniques")
    insights.append("\n • Stay hydrated and limit alcohol consumption")
    
    return "\n".join(insights)

def main():
    st.set_page_config(page_title="Health Risk Predictor", layout="wide", page_icon="🏥")
    
    # Custom CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        .main {
            background-color: #F3F4F6;
            font-family: 'Inter', sans-serif;
        }
        
        .stButton>button {
            background-color: #4F46E5;
            color: white;
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            margin: 0.5rem 0;
        }
        
        .stButton>button:hover {
            background-color: #4338CA;
            transform: translateY(-2px);
        }
        
        .metric-container {
            background-color: black;
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .sidebar {
            background-color: black;
            padding: 2rem;
            border-right: 1px solid #E5E7EB;
        }
        
        h1, h2, h3 {
            color: #ffffff;
            font-weight: 700;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Health Assessment Types")
        
        # Keep track of which assessment type is selected
        for pred_type, details in config['prediction_types'].items():
            if st.button(f"{details['icon']} {details['name']}", key=pred_type):
                st.session_state.current_prediction_type = pred_type
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
            This health risk predictor helps you understand your current health status 
            and provides personalized recommendations. Remember to consult healthcare 
            professionals for medical advice.
        """)
    
    # Main content
    if st.session_state.current_prediction_type:
        pred_type = st.session_state.current_prediction_type
        details = config['prediction_types'][pred_type]
        
        st.title(f"{details['icon']} {details['name']}")
        st.markdown(f"*{details['description']}*")
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("📊 Input Your Measurements")
            
            # Always show weight and height for BMI calculation
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
            height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70)
            bmi = calculate_bmi(weight, height)
            
            user_inputs = {'bmi': bmi}
            
            if "glucose_fasting" in details['required_metrics']:
                user_inputs['glucose_fasting'] = st.number_input(
                    "Fasting Glucose (mg/dL)",
                    min_value=70.0,
                    max_value=200.0,
                    value=95.0,
                    help="Measure after at least 8 hours of fasting"
                )
            
            if "glucose_post_meal" in details['required_metrics']:
                user_inputs['glucose_post_meal'] = st.number_input(
                    "Post-Meal Glucose (mg/dL)",
                    min_value=70.0,
                    max_value=300.0,
                    value=135.0,
                    help="Measure 2 hours after eating"
                )
            
            if "heart_rate" in details['required_metrics']:
                user_inputs['heart_rate'] = st.number_input(
                    "Heart Rate (bpm)",
                    min_value=40,
                    max_value=200,
                    value=75
                )
            
            if any(x in details['required_metrics'] for x in ['bp_systolic', 'bp_diastolic']):
                col_bp1, col_bp2 = st.columns(2)
                with col_bp1:
                    user_inputs['bp_systolic'] = st.number_input(
                        "Systolic BP (mmHg)",
                        min_value=70,
                        max_value=200,
                        value=120
                    )
                with col_bp2:
                    user_inputs['bp_diastolic'] = st.number_input(
                        "Diastolic BP (mmHg)",
                        min_value=40,
                        max_value=130,
                        value=80
                    )
        
        with col2:
            if st.button("Analyze Health Metrics", key="analyze"):
                st.markdown("### 📈 Analysis Results")
                
                # Create visualizations based on prediction type
                if pred_type == "diabetes":
                    col_g1, col_g2, col_g3 = st.columns(3)
                    
                    with col_g1:
                        fig_glucose = create_gauge_chart(
                            user_inputs['glucose_fasting'],
                            config['metrics']['glucose']['fasting']['ranges'],
                            "Fasting Glucose"
                        )
                        st.plotly_chart(fig_glucose, use_container_width=True)
                    
                    with col_g2:
                        fig_glucose_post = create_gauge_chart(
                            user_inputs['glucose_post_meal'],
                            config['metrics']['glucose']['post_meal']['ranges'],
                            "Post-Meal Glucose"
                        )
                        st.plotly_chart(fig_glucose_post, use_container_width=True)
                    
                    with col_g3:
                        fig_bmi = create_gauge_chart(
                            user_inputs['bmi'],
                            config['metrics']['bmi']['ranges'],
                            "BMI"
                        )
                        st.plotly_chart(fig_bmi, use_container_width=True)
                
                elif pred_type == "heart":
                    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                    with col_h1:
                        fig_hr = create_gauge_chart(
                            user_inputs['heart_rate'],
                            config['metrics']['vitals']['heart_rate']['ranges'],
                            "Heart Rate"
                        )
                        st.plotly_chart(fig_hr, use_container_width=True)
                    
                    with col_h2:
                        fig_bp_sys = create_gauge_chart(
                            user_inputs['bp_systolic'],
                            config['metrics']['vitals']['blood_pressure_systolic']['ranges'],
                            "Systolic BP"
                        )
                        st.plotly_chart(fig_bp_sys, use_container_width=True)
                    
                    with col_h3:
                        fig_bp_dia = create_gauge_chart(
                            user_inputs['bp_diastolic'],
                            config['metrics']['vitals']['blood_pressure_diastolic']['ranges'],
                            "Diastolic BP"
                        )
                        st.plotly_chart(fig_bp_dia, use_container_width=True)
                    
                    with col_h4:
                        fig_bmi = create_gauge_chart(
                            user_inputs['bmi'],
                            config['metrics']['bmi']['ranges'],
                            "BMI"
                        )
                        st.plotly_chart(fig_bmi, use_container_width=True)
                
                elif pred_type == "obesity":
                    col_o1, _ = st.columns(2)
                    with col_o1:
                        fig_bmi = create_gauge_chart(
                            user_inputs['bmi'],
                            config['metrics']['bmi']['ranges'],
                            "BMI"
                        )
                        st.plotly_chart(fig_bmi, use_container_width=True)
                
                # Get and display insights
                insights = get_health_insights(user_inputs, pred_type)
                
                st.markdown("### 🎯 Key Findings and Recommendations")
                st.markdown(f"""
                    <div class="metric-container">
                        {insights}
                    </div>
                """, unsafe_allow_html=True)
                
                # Display current date and time of analysis
                current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
                st.markdown(f"""
                    <div style='font-size: 0.8em; color: #666; margin-top: 1rem;'>
                        Analysis performed on {current_time}
                    </div>
                """, unsafe_allow_html=True)
                
                # Disclaimer
                st.markdown("---")
                st.markdown("""
                    <div style='background-color: black; padding: 1rem; border-radius: 10px; 
                              border-left: 4px solid #4F46E5; font-size: 0.9em;'>
                        <strong>Disclaimer:</strong> This tool provides general health insights and should not be used 
                        as a substitute for professional medical advice. Always consult with healthcare professionals 
                        for proper diagnosis and treatment.
                    </div>
                """, unsafe_allow_html=True)
    
    else:
        # Welcome screen when no assessment type is selected
        st.title("🏥 Welcome to Health Risk Predictor")
        st.markdown("""
            <div class="metric-container">
                <h3>Get Started</h3>
                <p>Select an assessment type from the sidebar to begin your health evaluation:</p>
                <ul>
                    <li>🩺 <strong>Diabetes Risk Assessment:</strong> Evaluate diabetes risk based on glucose levels</li>
                    <li>❤️ <strong>Heart Health Screening:</strong> Check cardiovascular health indicators</li>
                    <li>⚖️ <strong>Weight Status Analysis:</strong> Analyze BMI and related health factors</li>
                </ul>
                <p>Each assessment provides personalized insights and recommendations based on your measurements.</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
