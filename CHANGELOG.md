# Changelog

All notable changes to the ASTraM Mobility Digital Twin project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive `README.md` containing project architecture, modeling strategy, and setup instructions.

## [1.0.0] - 2026-06-16

### Added

- **Streamlit Dashboard App**:
  - `app.py`: Main layout for the Digital Twin Command Center.
  - `components.py`: Geographic Folium heatmap and Plotly trend visualizations.
  - `heuristics.py`: Rule-based engine mapping predictive severities to resource allocation (officers, tow trucks, barricades).
- **Machine Learning Models**:
  - `duration_model.json`: XGBoost Regressor forecasting incident duration in minutes.
  - `closure_model.json`: XGBoost Classifier estimating the probability of a carriageway closure.
  - Exported label encoders for Streamlit integration.
- **Jupyter Notebooks**:
  - `01_eda_and_cleaning.ipynb`: Cleaned UTC timestamps, imputed true resolution datetimes, clipped outliers to 48 hours, and dropped sparse data columns.
  - `02_model_training.ipynb`: Engineered 1/7/30-day historical corridor rolling counts, applied a strict chronological walk-forward data split, and computed evaluation metrics (AUC, MAE, Brier Score).
  - `03_osm_enrichment_optional.ipynb`: Included optional GeoPandas spatial join logic (using EPSG:32643) to enrich incident data with OpenStreetMap road classification tags.
- Setup file `requirements.txt` formatted for use with the `uv` package manager.

### Fixed

- Fixed a bug in `02_model_training.ipynb` where Pandas `rolling` dropped `NaT` indexing keys by explicitly filling `corridor` NAs before groupby.
- Resolved Streamlit absolute pathing issues (`__file__`) to allow the app to be run from any working directory safely without throwing "Models not found" errors.
- Resolved `streamlit_folium` deprecation warnings by replacing `folium_static` with `st_folium`.
- Cleared Plotly chart container width deprecation warnings.
