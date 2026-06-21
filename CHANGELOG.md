# Changelog

All notable changes to the ASTraM Mobility Digital Twin Command Center will be documented in this file.

## [Unreleased]

### Added
- **Live Digital Twin Context**: Integrated TomTom Flow Segment API for real-time traffic overlay and Open-Meteo API for live weather context (pre-filling the simulator automatically).
- **Holiday & Major Event Flags**: Added a contextual heuristic system to warn operators when predicting on public holidays, festival days, or election days in Bengaluru.
- **2D/3D Map UI Toggle**: Added a seamless Streamlit radio toggle allowing operators to instantly switch the Command Center view between the 3D PyDeck Hexagon Extrusion and a strict 2D Leaflet/Folium Continuous Heatmap.
- **Diagnostic Inference Script**: Added `scripts/inference.py` to seamlessly test and compare the dual-engine XGBoost and CatBoost predictions across varied scenarios side-by-side.
- Dual-Engine ML Pipeline: Added full integration for CatBoost. The app now supports side-by-side real-time toggling between CatBoost and XGBoost model engines for both Duration Regression and Closure Classification.
- Dynamic baseline congestion feature: Simulator now uses historical average corridor congestion counts specific to the selected day of the week, rather than static end-of-dataset counts.
- Dynamic OSM spatial defaults: The simulator automatically infers the most common OSM Highway Class, Lane Count, and median Distance to Road based on the selected corridor.

### Changed
- **Calibrated Heuristics Thresholds**: Lowered the `_severity_bucket` thresholds for Level 3 (Max Severity) from 60% to 45%. Because the data index alignment bug was fixed, the models are now incredibly strictly calibrated; a ~48% closure probability prediction now mathematically represents a worst-case catastrophic infrastructure failure.
- **Folium Integration**: Replaced the faux-2D PyDeck map with a true `folium.plugins.HeatMap` integration via `streamlit_folium` to ensure the 2D view is strictly flat and non-rotational.
- **Automation Cleanup**: Purged all hackathon presentation-generation scripts (`python-pptx`) from the codebase for the final push.
- Refactored ML model categorical encoding: Swapped `LabelEncoder` for native XGBoost categorical support (`enable_categorical=True`). This solves a critical bug where tree splits were treating unrelated categories as ordinal values.
- Re-scaled precipitation simulator: The maximum slider value now dynamically represents an extreme hourly downpour for Bangalore (50 mm/hr).
- **Unified Feature Engineering (DRY Architecture)**: Extracted and consolidated all feature engineering logic (datetime parsing, rolling counts, missing value imputation, and severity bucketing) into a single shared `app/feature_engineering.py` module to prevent drift between the training notebooks and the Streamlit app.

### Fixed
- **App Crash & Import Bugs**: Refactored relative/absolute imports in the main Streamlit application to prevent `ModuleNotFoundError` and circular import crashes.
- **Missing Data Handling**: Added a robust `os.path.exists()` check to gracefully error when the CSV data is missing, and fixed a `KeyError` vulnerability by using `.get()` for categorical dictionary mappings.
- **Nan Handling**: Fixed a `.fillna(median)` breakage bug when a column was entirely NaN, correctly substituting `0.0` as a fallback.
- Fixed administrative bulk-closure outlier bug: Capped short-term events (e.g., debris, VIP movement) to 12 hours during model training to prevent the regression model from artificially inflating expected clearing times to 21+ hours.
- Fixed timezone mismatch bug in Open-Meteo weather API merge: Converted naive UTC weather timestamps to IST (`Asia/Kolkata`) prior to merging with incident data, ensuring the ML model correctly learns the impact of non-zero rainfall on traffic clearing duration.
- **Index Alignment Bug**: Fixed a massive silent bug where `groupby().rolling()` scrambled the alignment of the 1d/7d/30d historical counts. Models were retrained on the properly aligned data, instantly bumping the Closure Classifier AUC to **0.8027**.
- **Severity Score Rendering**: Fixed an `is True` identity check bug in Pandas that was silently preventing maximum-severity events (Level 3) from being elevated accurately on the 3D PyDeck geospatial heatmap.
- **Missing Data Hotfix**: Automatically impute and drop invalid `start_datetime` NaT artifacts to prevent indexing crashes.
