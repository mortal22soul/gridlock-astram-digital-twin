import requests
import streamlit as st
import datetime

# WMO Weather interpretation codes (https://open-meteo.com/en/docs)
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Drizzle: Light", 53: "Drizzle: Moderate", 55: "Drizzle: Dense",
    56: "Freezing Drizzle: Light", 57: "Freezing Drizzle: Dense",
    61: "Rain: Slight", 63: "Rain: Moderate", 65: "Rain: Heavy",
    66: "Freezing Rain: Light", 67: "Freezing Rain: Heavy",
    71: "Snow fall: Slight", 73: "Snow fall: Moderate", 75: "Snow fall: Heavy",
    77: "Snow grains",
    80: "Rain showers: Slight", 81: "Rain showers: Moderate", 82: "Rain showers: Violent",
    85: "Snow showers slight", 86: "Snow showers heavy",
    95: "Thunderstorm: Slight or moderate",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
}

@st.cache_data(ttl=600)
def get_current_weather() -> dict | None:
    """Fetches real-time weather data for Bengaluru from Open-Meteo."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=12.9716&longitude=77.5946"
        "&current=temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m,weather_code"
        "&timezone=Asia%2FKolkata"
    )
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            curr = data.get("current", {})
            return {
                "temperature_c": curr.get("temperature_2m", 0.0),
                "precipitation_mm": curr.get("precipitation", 0.0),
                "rain_mm": curr.get("rain", 0.0),
                "humidity_percent": curr.get("relative_humidity_2m", 0),
                "wind_speed_kmh": curr.get("wind_speed_10m", 0.0),
                "weather_code": curr.get("weather_code", 0),
                "description": WMO_CODES.get(curr.get("weather_code", 0), "Unknown"),
                "timestamp": datetime.datetime.now().strftime("%I:%M %p")
            }
    except requests.exceptions.RequestException:
        pass
    return None
