import os
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env.local first, then .env fallback
load_dotenv(find_dotenv('.env.local'))
load_dotenv(find_dotenv('.env'))

@st.cache_data(ttl=300)
def get_traffic_flow(lat: float, lon: float) -> dict | None:
    """Fetches real-time traffic flow data from TomTom API."""
    api_key = os.environ.get("TOMTOM_API_KEY")

    if not api_key:
        return None
    
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&unit=KMPH&key={api_key}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            flow = data.get("flowSegmentData", {})
            current_speed = flow.get("currentSpeed")
            free_flow_speed = flow.get("freeFlowSpeed")
            
            if current_speed is not None and free_flow_speed is not None and free_flow_speed > 0:
                congestion_ratio = current_speed / free_flow_speed
                return {
                    "current_speed_kmh": current_speed,
                    "free_flow_speed_kmh": free_flow_speed,
                    "congestion_ratio": congestion_ratio,
                    "confidence": flow.get("confidence", 1.0)
                }
    except requests.exceptions.RequestException:
        pass
    
    return None

@st.cache_data
def get_corridor_centroid(df: pd.DataFrame, corridor: str) -> tuple[float, float] | None:
    """Finds the median lat/lon for a corridor from the historical data."""
    corridor_df = df[(df['corridor'] == corridor) & (df['latitude'].notna()) & (df['longitude'].notna())]
    if corridor_df.empty:
        return None
    
    return corridor_df['latitude'].median(), corridor_df['longitude'].median()
