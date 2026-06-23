import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import components
from heuristics import get_resource_recommendation
from feature_engineering import engineer_features
from traffic import get_traffic_flow, get_corridor_centroid
from weather import get_current_weather
from holidays import get_holiday_context
import os
import datetime

st.set_page_config(page_title="ASTraM Digital Twin Command Center", layout="wide", page_icon="🚦")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, '../data/augmented_astram_events.csv')


@st.cache_resource
def load_models():
    dur_xgb = xgb.XGBRegressor()
    dur_xgb.load_model(os.path.join(BASE_DIR, '../models/duration_model.json'))

    clos_xgb = xgb.XGBClassifier()
    clos_xgb.load_model(os.path.join(BASE_DIR, '../models/closure_model.json'))

    from catboost import CatBoostRegressor, CatBoostClassifier
    dur_cb = CatBoostRegressor()
    dur_cb.load_model(os.path.join(BASE_DIR, '../models/duration_model_cb.cbm'))

    clos_cb = CatBoostClassifier()
    clos_cb.load_model(os.path.join(BASE_DIR, '../models/closure_model_cb.cbm'))

    with open(os.path.join(BASE_DIR, '../models/category_mappings.json'), 'r') as f:
        cat_mappings = json.load(f)

    return dur_xgb, clos_xgb, dur_cb, clos_cb, cat_mappings


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return engineer_features(df)





def _compute_rolling_counts(df: pd.DataFrame, corridor: str, dayofweek: int) -> tuple[int, int, int]:
    """
    Return the historical average rolling counts for the selected corridor and day of week.
    This provides a much more accurate baseline congestion feature for the simulator than
    the static end-of-dataset approach.
    """
    subset = df[(df['corridor'] == corridor) & (df['start_datetime'].dt.dayofweek == dayofweek)]
    if subset.empty:
        subset = df[df['corridor'] == corridor]  # Fallback to corridor average across all days
    
    if subset.empty:
        return 0, 0, 0
    
    c_1d = int(subset['corridor_count_1d'].mean())
    c_7d = int(subset['corridor_count_7d'].mean())
    c_30d = int(subset['corridor_count_30d'].mean())
    return c_1d, c_7d, c_30d


def _severity_bucket(priority: str, clos_prob: float) -> int:
    """
    Map priority + continuous closure probability to a 0/1/2 severity bucket.

    Using the raw probability (rather than a hard 0.5 threshold) gives a more
    nuanced bucket for borderline cases, e.g. High-priority events near the
    closure-probability boundary stay at Medium instead of jumping to High.
    """
    if priority == 'High' and clos_prob >= 0.45:
        return 2   # High
    elif priority == 'High' or clos_prob >= 0.25:
        return 1   # Medium
    return 0       # Low


# ── Guard: models must exist before rendering anything else ──────────────────

if not os.path.exists(os.path.join(BASE_DIR, '../models/duration_model.json')) or not os.path.exists(os.path.join(BASE_DIR, '../models/duration_model_cb.cbm')):
    st.warning("Models not found. Please run the Jupyter notebooks in `notebooks/` first.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at `{DATA_PATH}`. Please run `notebooks/01_eda_and_cleaning.ipynb` first.")
    st.stop()

try:
    dur_xgb, clos_xgb, dur_cb, clos_cb, cat_mappings = load_models()
    df = load_data()
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

# ── Page header ──────────────────────────────────────────────────────────────

st.title("🚦 ASTraM Mobility Digital Twin Command Center")
st.markdown("Predictive layer for early incident mitigation and resource allocation.")

# ── Two-column layout ────────────────────────────────────────────────────────

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Historical Risk Heatmap")
    
    map_view = st.radio("Map View", ["3D Hexagon", "2D Heatmap"], horizontal=True, label_visibility="collapsed")
    if map_view == "3D Hexagon":
        deck = components.render_heatmap(df)
        st.pydeck_chart(deck)
    else:
        from streamlit_folium import st_folium
        m = components.render_2d_heatmap(df)
        st_folium(m, height=400, use_container_width=True, returned_objects=[])

    st.plotly_chart(components.plot_trend(df), width='stretch')

with col2:
    st.subheader("Incident Simulator")
    
    sim_date = st.date_input("Simulation Date", value=datetime.date.today())
    
    # --- LIVE CONTEXT ---
    weather_data = get_current_weather()
    holiday = get_holiday_context(sim_date)
    
    weather_success = weather_data is not None and not weather_data.get("error")
    
    if weather_success or holiday:
        st.markdown("##### 🌍 Live Context")
        if holiday:
            st.warning(f"⚠️ **Major Event Today:** {holiday}. Expect abnormal traffic patterns.")
        
        if weather_success:
            w_col1, w_col2, w_col3 = st.columns(3)
            w_col1.metric("Weather", weather_data["description"])
            w_col2.metric("Temp", f"{weather_data['temperature_c']}°C")
            w_col3.metric("Rain", f"{weather_data['precipitation_mm']} mm")
        st.markdown("---")
        
    if weather_data and weather_data.get("error"):
        st.warning(f"⚠️ Could not fetch live weather from Open-Meteo ({weather_data['error']}). Falling back to manual defaults.")
    # --------------------

    with st.form("simulator_form"):
        engine    = st.radio("Model Engine", ["CatBoost", "XGBoost"], horizontal=True)
        
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            cause     = st.selectbox("Event Cause", sorted(df['event_cause'].dropna().unique()))
        with f_col2:
            corridor  = st.selectbox("Corridor",    sorted(df['corridor'].dropna().unique()))
            
        priority  = st.selectbox("Priority", ['High', 'Low'])

        hour      = st.slider("Hour of Day", 0, 23, 12)
        
        f_col3, f_col4 = st.columns(2)
        with f_col3:
            day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            day_str   = st.selectbox("Day of Week", list(day_mapping.keys()))
            dayofweek = day_mapping[day_str]
        with f_col4:
            default_precip = float(weather_data['precipitation_mm']) if weather_success else 0.0
            precip    = st.number_input("Expected Rain (mm/hr) - Pre-filled with Live Data", min_value=0.0, max_value=50.0, value=default_precip, step=0.1)

        # Compute dynamic spatial defaults based on the selected corridor
        corridor_df = df[df['corridor'] == corridor]
        def_lanes = 2.0
        def_dist = 5.0
        def_class_idx = 0
        osm_options = sorted(df['osm_highway_class'].dropna().unique())

        if not corridor_df.empty:
            if not corridor_df['osm_lanes'].mode().empty:
                def_lanes = float(corridor_df['osm_lanes'].mode()[0])
            if not corridor_df['dist_to_nearest_road_m'].empty:
                def_dist = float(corridor_df['dist_to_nearest_road_m'].median())
            if not corridor_df['osm_highway_class'].mode().empty:
                osm_mode = corridor_df['osm_highway_class'].mode()[0]
                if osm_mode in osm_options:
                    def_class_idx = osm_options.index(osm_mode)

        with st.expander("OSM Properties", expanded=False):
            osm_class = st.selectbox("OSM Highway Class", osm_options, index=def_class_idx)
            osm_lanes = st.number_input("OSM Lanes", min_value=1.0, max_value=10.0, value=def_lanes, step=1.0)
            dist_road = st.number_input("Distance to Nearest Road (m)", min_value=0.0, value=def_dist, step=0.1)

        submitted = st.form_submit_button("Simulate Impact")

    if submitted:
        is_weekend = 1 if dayofweek in [5, 6] else 0

        # Rolling corridor counts dynamically adapt to historical day-of-week averages
        c_1d, c_7d, c_30d = _compute_rolling_counts(df, corridor, dayofweek)

        # Warn on unseen categorical values
        for field, value in [('event_cause', cause), ('corridor', corridor),
                              ('priority', priority), ('osm_highway_class', osm_class)]:
            if value not in cat_mappings.get(field, []):
                st.warning(f"'{value}' is not in the training vocabulary for **{field}**. "
                           "Prediction may be less accurate.")

        # Build a 1-row DataFrame with the proper category dtypes so that
        # XGBoost's native categorical support (enable_categorical=True)
        # recognises each cause/corridor/class as a distinct entity.
        feature_names = [
            'event_cause', 'corridor', 'priority',
            'hour', 'dayofweek', 'is_weekend',
            'corridor_count_1d', 'corridor_count_7d', 'corridor_count_30d',
            'precipitation_mm',
            'osm_highway_class', 'osm_lanes', 'dist_to_nearest_road_m'
        ]
        row = {
            'event_cause': cause, 'corridor': corridor, 'priority': priority,
            'hour': hour, 'dayofweek': dayofweek, 'is_weekend': is_weekend,
            'corridor_count_1d': float(c_1d), 'corridor_count_7d': float(c_7d),
            'corridor_count_30d': float(c_30d), 'precipitation_mm': precip,
            'osm_highway_class': osm_class, 'osm_lanes': osm_lanes,
            'dist_to_nearest_road_m': dist_road,
        }
        features_df = pd.DataFrame([row], columns=feature_names)

        # Apply the same CategoricalDtype used during XGBoost training
        # (CatBoost safely accepts and handles these natively as well)
        for col in ['event_cause', 'corridor', 'priority', 'osm_highway_class']:
            categories = cat_mappings.get(col, [])
            if not categories:
                st.warning(f"No category mapping found for **{col}** -- skipping dtype conversion.")
                continue
            cat_type = pd.CategoricalDtype(
                categories=categories, ordered=False
            )
            features_df[col] = features_df[col].astype(cat_type)

        # -- Predictions -------------------------------------------------------

        if engine == "CatBoost":
            clos_prob    = float(clos_cb.predict_proba(features_df)[0][1])
            dur_log      = float(dur_cb.predict(features_df)[0])
        else:
            clos_prob    = float(clos_xgb.predict_proba(features_df)[0][1])
            dur_log      = float(dur_xgb.predict(features_df)[0])

        dur_minutes  = float(np.expm1(dur_log))

        # Severity bucket uses continuous clos_prob, not a hard 0.5 threshold
        severity = _severity_bucket(priority, clos_prob)
        rec      = get_resource_recommendation(severity)

        # ── Results display ───────────────────────────────────────────────────

        st.markdown("### Prediction Results")
        
        # Display Live Traffic Context if available
        centroid = get_corridor_centroid(df, corridor)
        if centroid:
            traffic_data = get_traffic_flow(centroid[0], centroid[1])
            if traffic_data:
                components.render_traffic_badge(traffic_data)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Est. Duration",  f"{dur_minutes:.0f} mins")
        col_m2.metric("Closure Prob.",  f"{clos_prob * 100:.1f}%")
        col_m3.metric("Severity Level", rec['Level'])

        st.markdown("### Recommended Action")
        st.info(rec['Action'])

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("👮 Officers",   rec['Officers'])
        res_col2.metric("🚛 Tow Trucks", rec['Tow Trucks'])
        res_col3.metric("🚧 Barricades", rec['Barricades'])

        with st.expander("Feature values sent to model"):
            st.json({
                "event_cause":         cause,
                "corridor":            corridor,
                "priority":            priority,
                "hour":                hour,
                "day_of_week":         dayofweek,
                "is_weekend":          is_weekend,
                "corridor_count_1d":   c_1d,
                "corridor_count_7d":   c_7d,
                "corridor_count_30d":  c_30d,
                "precipitation_mm":    precip,
                "osm_highway_class":   osm_class,
                "osm_lanes":           osm_lanes,
                "dist_to_nearest_road_m": dist_road,
            })
