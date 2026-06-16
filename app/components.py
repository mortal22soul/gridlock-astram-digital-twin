import folium
from folium.plugins import HeatMap
import plotly.express as px
import pandas as pd

# Bounding box for Bengaluru — keeps rogue zero-coordinate rows off the map
_LAT_MIN, _LAT_MAX = 12.0, 13.5
_LON_MIN, _LON_MAX = 77.0, 78.0


def render_heatmap(df: pd.DataFrame) -> folium.Map:
    """Render a light-theme Folium heatmap weighted by severity_score."""
    m = folium.Map(
        location=[12.9716, 77.5946],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    clean_df = df.dropna(subset=['latitude', 'longitude'])
    clean_df = clean_df[
        (clean_df['latitude']  > _LAT_MIN) & (clean_df['latitude']  < _LAT_MAX) &
        (clean_df['longitude'] > _LON_MIN) & (clean_df['longitude'] < _LON_MAX)
    ]

    if clean_df.empty:
        return m

    # Vectorised — avoids slow iterrows() over thousands of rows
    heat_data = clean_df[['latitude', 'longitude', 'severity_score']].values.tolist()
    HeatMap(heat_data, radius=15, max_zoom=13).add_to(m)
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

    fig = px.line(
        trend,
        x='date',
        y='count',
        template="plotly_dark",
        title="Daily Incident Trend",
        labels={'date': 'Date', 'count': 'Incidents'},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=250)
    return fig
