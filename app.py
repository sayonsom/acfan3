import streamlit as st
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from typing import Dict, Union
import os

from predict import ComfortBasedPredictor  

def create_gauge_chart(value: float, title: str, min_val: float, max_val: float) -> go.Figure:
    """Create a gauge chart using plotly"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, (max_val-min_val)*0.33 + min_val], 'color': "lightgray"},
                {'range': [(max_val-min_val)*0.33 + min_val, (max_val-min_val)*0.66 + min_val], 'color': "gray"},
                {'range': [(max_val-min_val)*0.66 + min_val, max_val], 'color': "darkgray"}
            ]
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_energy_comparison_card(comfort_temp: float, predicted_temp: float):
    """Create a card showing energy savings comparison"""
    temp_difference = comfort_temp - predicted_temp
    print("Temperature difference:", temp_difference)
    
    # Calculate approximate energy savings (assumption: 5% energy saving per degree Celsius)
    energy_savings_percent = max(0, abs(temp_difference) * 5)
    
    st.markdown("### 💡 Energy Savings Analysis")
    st.text("Based on BEE Guidelines for AC usage [BEE is Bureau of Energy Efficiency, Govt. of India]")
    
    cols = st.columns([2, 1])
    with cols[0]:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h4 style='margin-top: 0;'>Temperature Comparison</h4>
            <p>Without AC-Fan Sync: Set AC at {:.1f}°C</p>
            <p>With AC-Fan Sync: Set AC at {:.1f}°C</p>
            <p>Temperature Optimization: {:.1f}°C</p>
            <h4>Estimated Energy Savings</h4>
            <p>Approximately {:.3f}% energy savings potential</p>
        </div>
        """.format(comfort_temp, predicted_temp, temp_difference, energy_savings_percent), unsafe_allow_html=True)
    
    with cols[1]:
        # Create a simple savings indicator
        fig = go.Figure(go.Indicator(
            mode="delta",
            value=predicted_temp,
            delta={'reference': comfort_temp, 'relative': True, 'valueformat': '.1f'},
            title={'text': "Temperature Difference"}
        ))
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="AC Fan Comfort Predictor", layout="wide")
    
    # Page title and description
    st.title("🌡️ AC Fan Comfort Predictor")
    st.markdown("""
    This app predicts optimal AC settings based on outdoor temperature and your comfort preferences.
    Adjust the sliders below to see how different conditions affect the recommended settings.
    """)
    
    # Initialize predictor
    MODEL_DIR = 'models'
    SCALER_DIR = 'scalers'
    
    predictor = ComfortBasedPredictor(
        model_path=os.path.join(MODEL_DIR, 'final_model.keras'),
        scaler_x_path=os.path.join(SCALER_DIR, 'scaler_X.joblib'),
        scaler_y_path=os.path.join(SCALER_DIR, 'scaler_y.joblib')
    )
    
    # Load the model
    if not predictor.load():
        st.error("Failed to load the model. Please check if model files exist in the correct location.")
        return
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environmental Conditions")
        outdoor_temp = st.slider(
            "Outdoor Temperature (°C)",
            min_value=float(predictor.input_bounds['outdoor_temp'][0]),
            max_value=float(predictor.input_bounds['outdoor_temp'][1]),
            value=30.0,
            step=0.5
        )
    
    with col2:
        st.subheader("Comfort Preferences")
        comfort_temp = st.slider(
            "Desired Comfort Temperature (°C)",
            min_value=float(predictor.input_bounds['comfort_temp'][0]),
            max_value=float(predictor.input_bounds['comfort_temp'][1]),
            value=24.0,
            step=0.5
        )
    
    # Make prediction
    result = predictor.predict_with_comfort(
        outdoor_temp=outdoor_temp,
        comfort_temp=comfort_temp
    )
    
    # Display results
    if result['success']:
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Create three columns for the gauge charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Temperature gauge
            fig_temp = create_gauge_chart(
                result['temperature'],
                "Suggested AC Setpoint (°C) using AI Model",
                18.0,
                30.0
            )
            st.plotly_chart(fig_temp, use_container_width=True)
            
        with col2:
            # Velocity gauge
            fig_vel = create_gauge_chart(
                result['velocity'],
                "Suggested Air Velocity (m/s) using AI Model",
                0.1,
                1.2
            )
            st.plotly_chart(fig_vel, use_container_width=True)
            
        with col3:
            # PMV gauge
            fig_pmv = create_gauge_chart(
                result['pmv'],
                "Predicted Mean Vote (PMV)",
                -3.0,
                3.0
            )
            st.plotly_chart(fig_pmv, use_container_width=True)
        
        # Display thermal sensation
        st.info(f"🌡️ Thermal Sensation: {result['thermal_sensation']}")
        
        # Add energy savings comparison
        st.markdown("---")
        create_energy_comparison_card(comfort_temp, result['temperature'])
        
    else:
        st.error(f"Prediction failed: {result['message']}")
    
    # Add footer with timestamp
    st.markdown("---")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()