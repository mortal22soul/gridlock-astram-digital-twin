import folium
from folium.plugins import HeatMap
import plotly.express as px
import pandas as pd

def render_heatmap(df):
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=11, tiles="cartodbdark_matter")
    # Drop rows with missing lat/long
    clean_df = df.dropna(subset=['latitude', 'longitude'])
    # Only keep reasonable latitudes and longitudes for Bengaluru (around 12-13, 77-78)
    clean_df = clean_df[(clean_df['latitude'] > 12) & (clean_df['latitude'] < 13.5) & 
                        (clean_df['longitude'] > 77) & (clean_df['longitude'] < 78)]
                        
    heat_data = [[row['latitude'], row['longitude'], row.get('severity_score', 1)] for index, row in clean_df.iterrows()]
    if heat_data:
        HeatMap(heat_data, radius=15, max_zoom=13).add_to(m)
    return m

def plot_trend(df):
    trend = df.groupby(df['start_datetime'].dt.date).size().reset_index(name='count')
    fig = px.line(trend, x='start_datetime', y='count', template="plotly_dark", title="Daily Incident Trend")
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=250)
    return fig
