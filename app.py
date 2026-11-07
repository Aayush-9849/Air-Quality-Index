"""
Streamlit Web Application for Real-Time Air Quality Index (AQI) Prediction
Uses trained Random Forest Regressor and standard CPCB/EPA thresholds.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_aqi.joblib"

st.set_page_config(
    page_title="Air Quality Index (AQI) Predictor",
    page_icon="🌫️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def get_aqi_badge(aqi):
    if aqi <= 50:
        return "Good", "#28a745", "Air quality is considered satisfactory, and air pollution poses little or no risk."
    elif aqi <= 100:
        return "Satisfactory", "#17a2b8", "Air quality is acceptable; however, some pollutants may cause moderate health concern for sensitive individuals."
    elif aqi <= 200:
        return "Moderate", "#ffc107", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif aqi <= 300:
        return "Poor", "#fd7e14", "Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects."
    elif aqi <= 400:
        return "Very Poor", "#dc3545", "Health alert: everyone may experience more serious health effects."
    else:
        return "Severe", "#6f42c1", "Health warning of emergency conditions. Entire population is likely affected."

def main():
    st.title("🌫️ Air Quality Index (AQI) Predictor")
    st.markdown(
        "Estimate the continuous **Air Quality Index (AQI)** from ambient pollutant concentrations using an ensemble **Random Forest Regressor** ($R^2 = 0.9958$)."
    )
    st.divider()

    model = load_model()

    col1, col2 = st.columns(2)

    with col1:
        pm25 = st.number_input("PM2.5 (Fine Particulate, µg/m³)", min_value=1.0, max_value=500.0, value=55.0, step=0.5)
        no2 = st.number_input("NO₂ (Nitrogen Dioxide, µg/m³)", min_value=1.0, max_value=400.0, value=45.0, step=0.5)
        co = st.number_input("CO (Carbon Monoxide, mg/m³)", min_value=0.1, max_value=50.0, value=2.8, step=0.1)

    with col2:
        pm10 = st.number_input("PM10 (Coarse Particulate, µg/m³)", min_value=2.0, max_value=600.0, value=110.0, step=1.0)
        so2 = st.number_input("SO₂ (Sulfur Dioxide, µg/m³)", min_value=1.0, max_value=500.0, value=22.0, step=0.5)
        o3 = st.number_input("O₃ (Ground-level Ozone, µg/m³)", min_value=1.0, max_value=400.0, value=48.0, step=0.5)

    if st.button("Predict Air Quality Index (AQI)", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3
        }])

        predicted_aqi = model.predict(input_data)[0]
        cat_name, color, advisory = get_aqi_badge(predicted_aqi)

        st.markdown(f"""
        <div style="background-color: {color}22; border-left: 6px solid {color}; padding: 18px; border-radius: 8px; margin-top: 20px;">
            <h2 style="margin: 0; color: {color};">Predicted AQI: {predicted_aqi:.1f}</h2>
            <h4 style="margin: 5px 0 10px 0; color: #333;">Air Quality Category: <strong>{cat_name}</strong></h4>
            <p style="margin: 0; color: #555; font-size: 14px;">{advisory}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Input Summary")
        st.dataframe(input_data)

if __name__ == "__main__":
    main()
