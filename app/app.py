import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import components
from heuristics import get_resource_recommendation
import os

st.set_page_config(page_title="ASTraM Digital Twin Command Center", layout="wide", page_icon="🚦")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, '../data/augmented_astram_events.csv')


@st.cache_resource
def load_models():
    duration_model = xgb.XGBRegressor()
    duration_model.load_model(os.path.join(BASE_DIR, '../models/duration_model.json'))

    closure_model = xgb.XGBClassifier()
    closure_model.load_model(os.path.join(BASE_DIR, '../models/closure_model.json'))

    with open(os.path.join(BASE_DIR, '../models/category_mappings.json'), 'r') as f:
        cat_mappings = json.load(f)

    return duration_model, closure_model, cat_mappings


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], utc=True)
    df = df.dropna(subset=['start_datetime'])
    df = df.sort_values('start_datetime').reset_index(drop=True)

    # Recreate the rolling counts features here since they aren't saved in the CSV
    df['corridor'] = df['corridor'].fillna('Unknown')
    df_temp = df.set_index('start_datetime')
    rolling_1d = df_temp.groupby('corridor', dropna=False).rolling('1D')['id'].count().reset_index(name='count_1d')
    rolling_7d = df_temp.groupby('corridor', dropna=False).rolling('7D')['id'].count().reset_index(name='count_7d')
    rolling_30d = df_temp.groupby('corridor', dropna=False).rolling('30D')['id'].count().reset_index(name='count_30d')

    df['corridor_count_1d'] = rolling_1d['count_1d'].values - 1
    df['corridor_count_7d'] = rolling_7d['count_7d'].values - 1
    df['corridor_count_30d'] = rolling_30d['count_30d'].values - 1

    # Derive a severity_score for the heatmap weight:
    # High priority + closure → 3, High priority only → 2, Low → 1
    if 'severity_score' not in df.columns:
        conditions = [
            (df['priority'] == 'High') & (df['requires_road_closure'] is True),
            (df['priority'] == 'High'),
        ]
        df['severity_score'] = np.select(conditions, [3, 2], default=1)
    return df





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
    if priority == 'High' and clos_prob >= 0.6:
        return 2   # High
    elif priority == 'High' or clos_prob >= 0.4:
        return 1   # Medium
    return 0       # Low


# ── Guard: models must exist before rendering anything else ──────────────────

if not os.path.exists(os.path.join(BASE_DIR, '../models/duration_model.json')):
    st.warning("Models not found. Please run the Jupyter notebooks in `notebooks/` first.")
    st.stop()

try:
    dur_model, clos_model, cat_mappings = load_models()
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
    st.subheader("Historical Risk Heatmap (3D Hexagon)")
    deck = components.render_heatmap(df)
    st.pydeck_chart(deck)

    st.plotly_chart(components.plot_trend(df), width='stretch')

with col2:
    st.subheader("Incident Simulator")
    with st.form("simulator_form"):
        cause     = st.selectbox("Event Cause", sorted(df['event_cause'].dropna().unique()))
        corridor  = st.selectbox("Corridor",    sorted(df['corridor'].dropna().unique()))
        priority  = st.selectbox("Priority", ['High', 'Low'])

        hour      = st.slider("Hour of Day", 0, 23, 12)
        day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        day_str   = st.selectbox("Day of Week", list(day_mapping.keys()))
        dayofweek = day_mapping[day_str]
        precip    = st.slider("Expected Rain (mm/hr)", 0.0, 50.0, 0.0)

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

        # Apply the same CategoricalDtype used during training
        for col in ['event_cause', 'corridor', 'priority', 'osm_highway_class']:
            cat_type = pd.CategoricalDtype(
                categories=cat_mappings[col], ordered=False
            )
            features_df[col] = features_df[col].astype(cat_type)

        # -- Predictions -------------------------------------------------------

        clos_prob    = float(clos_model.predict_proba(features_df)[0][1])
        dur_log      = float(dur_model.predict(features_df)[0])
        dur_minutes  = float(np.expm1(dur_log))

        # Severity bucket uses continuous clos_prob, not a hard 0.5 threshold
        severity = _severity_bucket(priority, clos_prob)
        rec      = get_resource_recommendation(severity)

        # ── Results display ───────────────────────────────────────────────────

        st.markdown("### Prediction Results")
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
