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

# Prefer enriched dataset (has OSM columns); fall back to cleaned if not present.
_ENRICHED = os.path.join(BASE_DIR, '../data/enriched_astram_events.csv')
_CLEANED  = os.path.join(BASE_DIR, '../data/cleaned_astram_events.csv')
DATA_PATH = _ENRICHED if os.path.exists(_ENRICHED) else _CLEANED


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
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], utc=True)

    # Derive a severity_score for the heatmap weight:
    # High priority + closure → 3, High priority only → 2, Low → 1
    if 'severity_score' not in df.columns:
        conditions = [
            (df['priority'] == 'High') & (df['requires_road_closure'] == True),
            (df['priority'] == 'High'),
        ]
        df['severity_score'] = np.select(conditions, [3, 2], default=1)
    return df


def _encode(encoders: dict, field: str, value: str) -> int:
    """Encode a categorical value, returning -1 and surfacing a warning if unseen."""
    le = encoders[field]
    if value in le.classes_:
        return int(le.transform([value])[0])
    st.warning(f"'{value}' is not in the training vocabulary for **{field}**. "
               "Prediction may be less accurate.")
    # Fall back to the most-frequent class (index 0 after label-encoding is arbitrary,
    # so we use the class that appears at position 0 in the sorted encoder classes).
    return 0


def _compute_rolling_counts(df: pd.DataFrame, corridor: str) -> tuple[int, int, int]:
    """
    Approximate the corridor rolling-count features from the loaded historical data.

    These features were present during training; passing zeros causes systematic
    under-prediction on busy corridors.  Using historical counts is the closest
    approximation available at inference time without a live event stream.
    """
    corridor_df = df[df['corridor'] == corridor]
    latest = df['start_datetime'].max()  # treat most-recent event as 'now'

    c_1d  = int((corridor_df['start_datetime'] > latest - pd.Timedelta('1D')).sum())
    c_7d  = int((corridor_df['start_datetime'] > latest - pd.Timedelta('7D')).sum())
    c_30d = int((corridor_df['start_datetime'] > latest - pd.Timedelta('30D')).sum())
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
    dur_model, clos_model, encoders = load_models()
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
    m = components.render_heatmap(df)
    st_folium(m, width=800, height=500, returned_objects=[])

    st.plotly_chart(components.plot_trend(df), width='stretch')

with col2:
    st.subheader("Incident Simulator")
    with st.form("simulator_form"):
        cause     = st.selectbox("Event Cause", sorted(df['event_cause'].dropna().unique()))
        corridor  = st.selectbox("Corridor",    sorted(df['corridor'].dropna().unique()))
        priority  = st.selectbox("Priority", ['High', 'Low'])

        hour      = st.slider("Hour of Day",               0, 23, 12)
        dayofweek = st.slider("Day of Week (0=Mon, 6=Sun)", 0,  6,  0)

        submitted = st.form_submit_button("Simulate Impact")

    if submitted:
        is_weekend = 1 if dayofweek in [5, 6] else 0

        # Rolling corridor counts — computed from historical data rather than
        # hardcoded to 0, which would degrade predictions on busy corridors.
        c_1d, c_7d, c_30d = _compute_rolling_counts(df, corridor)

        # Encode categoricals with graceful fallback for unseen labels
        c_cause = _encode(encoders, 'event_cause', cause)
        c_corr  = _encode(encoders, 'corridor',    corridor)
        c_prio  = _encode(encoders, 'priority',    priority)

        features = np.array([[c_cause, c_corr, c_prio,
                               hour, dayofweek, is_weekend,
                               c_1d, c_7d, c_30d]])

        # ── Predictions ──────────────────────────────────────────────────────

        clos_prob    = float(clos_model.predict_proba(features)[0][1])
        dur_log      = float(dur_model.predict(features)[0])
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
            })
