# Changelog

All notable changes to the ASTraM Mobility Digital Twin Command Center will be documented in this file.

## [Unreleased]

### Added
- **Diagnostic Inference Script**: Added `scripts/inference.py` to seamlessly test and compare the dual-engine XGBoost and CatBoost predictions across varied scenarios side-by-side.
- Dual-Engine ML Pipeline: Added full integration for CatBoost. The app now supports side-by-side real-time toggling between CatBoost and XGBoost model engines for both Duration Regression and Closure Classification.
- Dynamic baseline congestion feature: Simulator now uses historical average corridor congestion counts specific to the selected day of the week, rather than static end-of-dataset counts.
- Dynamic OSM spatial defaults: The simulator automatically infers the most common OSM Highway Class, Lane Count, and median Distance to Road based on the selected corridor.

### Changed
- Refactored ML model categorical encoding: Swapped `LabelEncoder` for native XGBoost categorical support (`enable_categorical=True`). This solves a critical bug where tree splits were treating unrelated categories as ordinal values.
- Re-scaled precipitation simulator: The maximum slider value now dynamically represents an extreme hourly downpour for Bangalore (50 mm/hr).
- **Unified Feature Engineering (DRY Architecture)**: Extracted and consolidated all feature engineering logic (datetime parsing, rolling counts, missing value imputation, and severity bucketing) into a single shared `app/feature_engineering.py` module to prevent drift between the training notebooks and the Streamlit app.

### Fixed
- Fixed administrative bulk-closure outlier bug: Capped short-term events (e.g., debris, VIP movement) to 12 hours during model training to prevent the regression model from artificially inflating expected clearing times to 21+ hours.
- Fixed timezone mismatch bug in Open-Meteo weather API merge: Converted naive UTC weather timestamps to IST (`Asia/Kolkata`) prior to merging with incident data, ensuring the ML model correctly learns the impact of non-zero rainfall on traffic clearing duration.
- **Index Alignment Bug**: Fixed a massive silent bug where `groupby().rolling()` scrambled the alignment of the 1d/7d/30d historical counts. Models were retrained on the properly aligned data, instantly bumping the Closure Classifier AUC to **0.8027**.
- **Severity Score Rendering**: Fixed an `is True` identity check bug in Pandas that was silently preventing maximum-severity events (Level 3) from being elevated accurately on the 3D PyDeck geospatial heatmap.
- **Missing Data Hotfix**: Automatically impute and drop invalid `start_datetime` NaT artifacts to prevent indexing crashes.
