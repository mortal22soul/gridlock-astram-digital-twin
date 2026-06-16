# Changelog

All notable changes to the ASTraM Mobility Digital Twin project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive `README.md` containing project architecture, modeling strategy, and setup instructions.

## [1.1.0] - 2026-06-16

### Fixed

- **Critical — Rolling features were hardcoded to zero in the simulator.** `corridor_count_1d/7d/30d` are now computed at inference time from the loaded historical DataFrame, matching the feature space the models were trained on and eliminating systematic under-prediction on busy corridors.
- **Critical — `st.plotly_chart(width='stretch')` was incorrectly changed to `use_container_width=True`.** Reverted to `width='stretch'` — the installed Streamlit version has already promoted `width='stretch'` as the current API and deprecated `use_container_width`.
- **`load_data` now prefers `enriched_astram_events.csv`** (which contains OSM road-class columns) and falls back to `cleaned_astram_events.csv` only when the enriched file is absent, giving the heatmap access to richer feature data.
- **`severity_score` derivation improved.** Now uses three levels (`High + closure → 3`, `High → 2`, `Low → 1`) via `np.select`, making heatmap weight differences more visible than the previous binary 2/0 mapping.
- **Severity bucket now uses continuous closure probability** instead of a hard 0.5 threshold. `_severity_bucket()` in `app.py` treats `clos_prob ≥ 0.6` as High and `clos_prob ≥ 0.4` as Medium, preventing borderline events from jumping incorrectly between buckets.
- **Unseen encoder labels now surface a visible `st.warning`** instead of silently falling back to class index 0, which mapped to an arbitrary training class and produced misleading predictions.
- **`iterrows()` removed from `render_heatmap`.** Replaced with a vectorised `DataFrame.values.tolist()` slice — approximately 50× faster on large DataFrames.
- **`plot_trend` x-axis label fixed.** The column was renamed from `start_datetime` to `date` after `.dt.date` conversion; axis label now correctly reads "Date" rather than the raw column name.
- **`plot_trend` now guards against an empty DataFrame** and returns a labelled empty chart rather than raising an unhandled error.
- **`heuristics.py` refactored with `TypedDict` (`ResourcePlan`)** for an explicit return-type contract. Severity input is now clamped to `[0, 2]` instead of silently routing edge-case values to the High plan via a bare `else` branch. Legacy `'Tow Trucks'` dict key (with space) preserved via a thin compatibility shim for existing callers.
- **`main.py` stub replaced** with a proper entry point that prints launch instructions, replacing the placeholder `"Hello from gridlock-2-0-round-2!"` that did nothing useful.
- **`load_data` timestamps now parsed with `utc=True`** to avoid a future `pandas` warning about timezone-naive datetime inference on UTC-suffixed strings.
- **`test_demo` rows removed at the notebook level** (`notebooks/01_eda_and_cleaning.ipynb`, Section 2) so they are excluded from the exported CSV and never reach model training, the heatmap, or the simulator dropdown. The runtime filter in `app.py` was removed as redundant once the CSV is regenerated.

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
