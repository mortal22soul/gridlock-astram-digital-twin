import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from streamlit_folium import st_folium
import components
from heuristics import get_resource_recommendation
import os

st.set_page_config(page_title="ASTraM Digital Twin Command Center", layout="wide", page_icon="🚦")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    duration_model = xgb.XGBRegressor()
    duration_model.load_model(os.path.join(BASE_DIR, '../models/duration_model.json'))
    
    closure_model = xgb.XGBClassifier()
    closure_model.load_model(os.path.join(BASE_DIR, '../models/closure_model.json'))
    
    with open(os.path.join(BASE_DIR, '../models/label_encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
        
    return duration_model, closure_model, encoders

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, '../data/cleaned_astram_events.csv'))
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    # For heatmap, approximate severity_score if missing
    if 'severity_score' not in df.columns:
        df['severity_score'] = np.where(df['priority'] == 'High', 2, 0)
    return df

st.title("🚦 ASTraM Mobility Digital Twin Command Center")
st.markdown("Predictive layer for early incident mitigation and resource allocation.")

if not os.path.exists(os.path.join(BASE_DIR, '../models/duration_model.json')):
    st.warning("Models not found. Please run the Jupyter notebooks in `notebooks/` first.")
    st.stop()

try:
    dur_model, clos_model, encoders = load_models()
    df = load_data()
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Historical Risk Heatmap")
    m = components.render_heatmap(df)
    st_folium(m, width=800, height=500, returned_objects=[])
    
    st.plotly_chart(components.plot_trend(df), width='stretch')

with col2:
    st.subheader("Incident Simulator")
    with st.form("simulator_form"):
        cause = st.selectbox("Event Cause", df['event_cause'].dropna().unique())
        corridor = st.selectbox("Corridor", df['corridor'].dropna().unique())
        priority = st.selectbox("Priority", ['High', 'Low'])
        
        hour = st.slider("Hour of Day", 0, 23, 12)
        dayofweek = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
        
        submitted = st.form_submit_button("Simulate Impact")
        
    if submitted:
        is_weekend = 1 if dayofweek in [5, 6] else 0
        
        # Approximate rolling features based on historical median for the corridor (for simplicity in demo)
        c_1d = 0; c_7d = 0; c_30d = 0
        
        # Encode
        c_cause = encoders['event_cause'].transform([cause])[0] if cause in encoders['event_cause'].classes_ else 0
        c_corr = encoders['corridor'].transform([corridor])[0] if corridor in encoders['corridor'].classes_ else 0
        c_prio = encoders['priority'].transform([priority])[0] if priority in encoders['priority'].classes_ else 0
        
        features = np.array([[c_cause, c_corr, c_prio, hour, dayofweek, is_weekend, c_1d, c_7d, c_30d]])
        
        # Predict Closure
        clos_prob = clos_model.predict_proba(features)[0][1]
        requires_closure = bool(clos_prob > 0.5)
        
        # Predict Duration
        dur_log = dur_model.predict(features)[0]
        dur_minutes = np.expm1(dur_log)
        
        # Severity Bucket
        if priority == 'High' and requires_closure:
            severity = 2
        elif priority == 'High':
            severity = 1
        else:
            severity = 0
            
        rec = get_resource_recommendation(severity)
        
        st.markdown("### Prediction Results")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Est. Duration", f"{dur_minutes:.0f} mins")
        col_m2.metric("Closure Prob.", f"{clos_prob*100:.1f}%")
        
        st.markdown("### Recommended Action")
        st.info(rec['Action'])
        st.write(f"**Officers Needed:** {rec['Officers']}")
        st.write(f"**Tow Trucks:** {rec['Tow Trucks']}")
        st.write(f"**Barricades:** {rec['Barricades']}")
