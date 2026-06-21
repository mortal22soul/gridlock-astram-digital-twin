import pydeck as pdk
import plotly.express as px
import pandas as pd

# Bounding box for Bengaluru — keeps rogue zero-coordinate rows off the map
_LAT_MIN, _LAT_MAX = 12.0, 13.5
_LON_MIN, _LON_MAX = 77.0, 78.0


def render_heatmap(df: pd.DataFrame) -> pdk.Deck:
    """Render a dark-theme PyDeck 3D Hexagon heatmap weighted by severity_score."""
    clean_df = df.dropna(subset=['latitude', 'longitude'])
    clean_df = clean_df[
        (clean_df['latitude']  > _LAT_MIN) & (clean_df['latitude']  < _LAT_MAX) &
        (clean_df['longitude'] > _LON_MIN) & (clean_df['longitude'] < _LON_MAX)
    ]

    if clean_df.empty:
        # Return an empty map centered on Bengaluru
        return pdk.Deck(
            map_style='dark',
            initial_view_state=pdk.ViewState(
                latitude=12.9716,
                longitude=77.5946,
                zoom=11,
                pitch=50,
            )
        )

    # Define the PyDeck HexagonLayer
    layer = pdk.Layer(
        'HexagonLayer',
        data=clean_df,
        get_position='[longitude, latitude]',
        radius=40,
        coverage=0.8,
        opacity=0.6,
        elevation_scale=8,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,
        get_elevation_weight='severity_score',
    )

    # Set the viewport location
    view_state = pdk.ViewState(
        latitude=12.9716,
        longitude=77.5946,
        zoom=11,
        pitch=50,
    )

    # Render
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style='dark',
        tooltip={"text": "Incidents"}
    )
    return deck


def render_2d_heatmap(df: pd.DataFrame):
    """Render a standard Leaflet (Folium) 2D Heatmap."""
    import folium
    from folium.plugins import HeatMap

    clean_df = df.dropna(subset=['latitude', 'longitude'])
    clean_df = clean_df[
        (clean_df['latitude']  > _LAT_MIN) & (clean_df['latitude']  < _LAT_MAX) &
        (clean_df['longitude'] > _LON_MIN) & (clean_df['longitude'] < _LON_MAX)
    ]

    m = folium.Map(location=[12.9716, 77.5946], zoom_start=11, tiles="CartoDB dark_matter")
    if not clean_df.empty:
        heat_data = clean_df[['latitude', 'longitude', 'severity_score']].values.tolist()
        HeatMap(heat_data, radius=15, blur=10).add_to(m)
    
    return m


def plot_trend(df: pd.DataFrame):
    """Return a Plotly line chart of daily incident counts."""
    if df.empty or 'start_datetime' not in df.columns:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            title="Daily Incident Trend (no data)",
            template="plotly_dark",
            height=250,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        return fig

    trend = (
        df.groupby(df['start_datetime'].dt.date)
        .size()
        .reset_index(name='count')
        .rename(columns={'start_datetime': 'date'})
    )

    # Calculate 7-day moving average
    trend['7-Day MA'] = trend['count'].rolling(window=7, min_periods=1).mean()

    fig = px.line(
        trend,
        x='date',
        y=['count', '7-Day MA'],
        template="plotly_dark",
        title="Daily Incident Trend",
        labels={'date': 'Date', 'value': 'Incidents', 'variable': 'Metric'},
        color_discrete_map={'count': '#636EFA', '7-Day MA': '#EF553B'}
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=250, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig

def render_traffic_badge(traffic_data: dict):
    """Render a colored badge for live traffic congestion."""
    import streamlit as st
    ratio = traffic_data["congestion_ratio"]
    curr = traffic_data["current_speed_kmh"]
    free = traffic_data["free_flow_speed_kmh"]
    
    if ratio < 0.5:
        color = "#ff4b4b" # Red
        status = "Heavy Congestion"
    elif ratio < 0.8:
        color = "#ffa421" # Orange
        status = "Moderate Traffic"
    else:
        color = "#00c853" # Green
        status = "Free Flowing"
        
    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 5px; background-color: {color}20; border: 1px solid {color}; margin-bottom: 10px;">
            <strong style="color: {color}; font-size: 1.1em;">🚦 Live Traffic: {status}</strong><br/>
            Current Speed: <b>{curr} km/h</b> (Free flow: {free} km/h)
        </div>
        """,
        unsafe_allow_html=True
    )
