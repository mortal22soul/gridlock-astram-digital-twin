import pandas as pd
import numpy as np
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.feature_engineering import engineer_features

# We can import models directly via catboost and xgboost, or use app's loader
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier

def load_models():
    dur_xgb = xgb.XGBRegressor()
    dur_xgb.load_model('models/duration_model.json')

    clos_xgb = xgb.XGBClassifier()
    clos_xgb.load_model('models/closure_model.json')

    dur_cb = CatBoostRegressor()
    dur_cb.load_model('models/duration_model_cb.cbm')

    clos_cb = CatBoostClassifier()
    clos_cb.load_model('models/closure_model_cb.cbm')

    with open('models/category_mappings.json', 'r') as f:
        cat_mappings = json.load(f)

    return dur_xgb, clos_xgb, dur_cb, clos_cb, cat_mappings

def main():
    print("Loading models...")
    dur_xgb, clos_xgb, dur_cb, clos_cb, cat_mappings = load_models()

    print("Running Inference Comparison...\n")

    # Define some interesting test cases to highlight differences
    test_cases = [
        # Typical Event
        {"event_cause": "vehicle_breakdown", "priority": "Low", "precipitation_mm": 0.0, "desc": "Typical breakdown (Dry)"},
        # High Variance/Sparsity Event
        {"event_cause": "Debris", "priority": "High", "precipitation_mm": 0.0, "desc": "Rare event (Debris, Dry)"},
        # Weather impact
        {"event_cause": "vehicle_breakdown", "priority": "Low", "precipitation_mm": 10.0, "desc": "Typical breakdown (Heavy Rain)"},
        # Infrastructure Failure
        {"event_cause": "tree_fall", "priority": "High", "precipitation_mm": 20.0, "desc": "Tree fall (Heavy Rain)"},
        # Admin / Planned Closure
        {"event_cause": "construction", "priority": "Low", "precipitation_mm": 0.0, "desc": "Construction (Dry)"},
    ]

    base_params = {
        'corridor': 'Old Airport Road',
        'hour': 14,
        'dayofweek': 2,
        'is_weekend': 0,
        'corridor_count_1d': 2.0,
        'corridor_count_7d': 15.0,
        'corridor_count_30d': 50.0,
        'osm_highway_class': 'primary',
        'osm_lanes': 3.0,
        'dist_to_nearest_road_m': 0.0
    }

    feature_names = [
        'event_cause', 'corridor', 'priority', 'hour', 'dayofweek',
        'is_weekend', 'corridor_count_1d', 'corridor_count_7d',
        'corridor_count_30d', 'precipitation_mm', 'osm_highway_class',
        'osm_lanes', 'dist_to_nearest_road_m'
    ]

    # Print header
    print(f"{'Scenario':<35} | {'XGB Duration':<15} | {'CB Duration':<15} | {'XGB Close %':<15} | {'CB Close %':<15}")
    print("-" * 105)

    for case in test_cases:
        row = base_params.copy()
        row.update({
            'event_cause': case['event_cause'],
            'priority': case['priority'],
            'precipitation_mm': case['precipitation_mm']
        })
        
        df = pd.DataFrame([row], columns=feature_names)
        
        # Apply categorical dtypes as required by XGBoost natively
        for col in ['event_cause', 'corridor', 'priority', 'osm_highway_class']:
            cat_type = pd.CategoricalDtype(categories=cat_mappings[col], ordered=False)
            df[col] = df[col].astype(cat_type)

        # Predict
        xgb_dur = np.expm1(dur_xgb.predict(df)[0])
        xgb_cls = clos_xgb.predict_proba(df)[0][1] * 100

        cb_dur = np.expm1(dur_cb.predict(df)[0])
        cb_cls = clos_cb.predict_proba(df)[0][1] * 100

        print(f"{case['desc']:<35} | {xgb_dur:<15.1f} | {cb_dur:<15.1f} | {xgb_cls:<14.1f}% | {cb_cls:<14.1f}%")

if __name__ == "__main__":
    main()
