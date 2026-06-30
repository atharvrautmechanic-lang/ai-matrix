import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import requests
from dotenv import load_dotenv

# Load secure tokens if available
load_dotenv()

st.set_page_config(page_title="Satellite Sensor Matrix", page_icon="🛰️", layout="wide")

st.title("🛰️ Data-Fused Virtual Satellite Sensor Matrix")
st.caption("Enterprise Environmental Proxy Architecture — Powered by Machine Learning")
st.markdown("---")

MODEL_BIN = "multi_pollutant_forest.pkl"
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

def get_live_elevation(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        return requests.get(url, timeout=2).json()['elevation'][0]
    except Exception:
        return 210.0

st.sidebar.header("🕹️ Target Matrix Control Panel")
target_lat = st.sidebar.number_input("Target Latitude Float", value=28.6139, format="%.4f")
target_lon = st.sidebar.number_input("Target Longitude Float", value=77.2090, format="%.4f")
simulated_aod = st.sidebar.slider("Simulated Aerosol Optical Depth (AOD)", 0.0, 2.0, 0.35)

if st.sidebar.button("⚡ Run Live System Inference"):
    # If model matrix asset file hasn't been uploaded yet, use structural fallback equations for demo purposes
    if not os.path.exists(MODEL_BIN):
        st.warning("⚠️ Running in Algorithmic Proxy Mode (Model matrix .pkl asset not found).")
        resolved_elevation = get_live_elevation(target_lat, target_lon)
        
        # Algorithmic fallback simulations
        pm25 = max(15.0, (simulated_aod * 180.0) - (resolved_elevation * 0.05))
        pm10 = pm25 * 1.6
        no2 = 22.4 + (simulated_aod * 12.0)
        so2 = 4.5 + (simulated_aod * 2.1)
        co = 0.35 + (simulated_aod * 0.4)
        hcho = 0.00015 + (simulated_aod * 0.0003)
        
        predictions = [pm25, pm10, no2, so2, co, hcho]
    else:
        with st.spinner("Processing machine learning weights..."):
            model = joblib.load(MODEL_BIN)
            resolved_elevation = get_live_elevation(target_lat, target_lon)
            mock_feature_vector = [
                target_lat, target_lon, resolved_elevation, 6.5, 
                298.15, 0.5, -0.8, 0.94, simulated_aod / (resolved_elevation + 1), 
                simulated_aod, 0.0, 1.2e-5, 4.3e-6, 0.002, 2.1e-5
            ]
            predictions = model.predict(np.array([mock_feature_vector]))[0]
            st.success("Inference completed via Random Forest Brain Array.")
            
    col1, col2, col3 = st.columns(3)
    col1.metric("PM2.5 Proxy Sub-Index", f"{predictions[0]:.2f} AQI")
    col2.metric("Ground PM10 Level", f"{predictions[1]:.2f} µg/m³")
    col3.metric("Ground NO2 Density", f"{predictions[2]:.2f} µg/m³")
    
    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    col4.metric("Ground SO2 Density", f"{predictions[3]:.2f} µg/m³")
    col5.metric("Ground CO Density", f"{predictions[4]:.2f} mg/m³")
    col6.metric("Predicted HCHO Gas Layer", f"{predictions[5]:.6f} mol/m²")
else:
    st.info("👈 Set coordinates and select 'Run Live System Inference' in the sidebar to simulate.")
