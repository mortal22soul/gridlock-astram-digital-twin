# 🚦 ASTraM Mobility Digital Twin Command Center

> **Gridlock 2.0 Hackathon Submission**  
> **Problem Statement 2:** Event-Driven Congestion Mitigation

![ASTraM Command Center Concept](https://img.shields.io/badge/Status-Prototype_Complete-success?style=for-the-badge)

## 📖 Overview

The **ASTraM Mobility Digital Twin Command Center** is a predictive intelligence system designed to fill a critical gap identified in the Bengaluru Traffic Police's (BTP) Digital Twin tender: the ability to pre-emptively mitigate the impact of traffic incidents.

Instead of relying heavily on expensive, delayed camera-feed analyses, this prototype leverages historical incident data (logged natively by ASTraM) to train machine learning models. The system instantly forecasts the **Duration** and **Road Closure Probability** of an incident the moment it is logged, and maps those predictions to an actionable **Resource Allocation Heuristic** (recommending officers, barricades, and tow trucks).

## ✨ Key Features

- **Modeling:** Trains dual sets of models (XGBoost and CatBoost) so you can toggle between engines in real-time. A standalone `scripts/inference.py` utility is included for head-to-head testing.
  - `Classifier`: Predicts the probability that an event requires a road closure. Achieves a powerful **0.80+ AUC**.
  - `Regressor`: Predicts the exact duration (in minutes) to clear the incident. Features an incredible **~20-30 minute Median Absolute Error** for standard acute emergencies (like vehicle breakdowns and accidents), while correctly modeling the heavy-tailed variance of 48-hour infrastructure failures (like water logging and deep potholes).
  - *Note:* Both models natively support categorical groupings via Pandas `category` dtypes or native CatBoost features, and they share a strictly unified DRY feature engineering pipeline (`app/feature_engineering.py`).
- **Rule-Based Resource Engine:** Translates predicted severity buckets into actionable ground deployments so the Control Room can act instantly.
- **Geospatial Risk Heatmap:** Visualizes historical hotspots across the Bengaluru road network using Folium.
- **OSM & Weather Enrichment:** Automatically fetches hourly precipitation from Open-Meteo and incorporates OpenStreetMap road infrastructure (e.g., highway class, lanes) via a rapid `GeoPandas` nearest-neighbor spatial join for enhanced modeling signal.
- **Efficient Preprocessing:** Handles missing values and merges historical weather data (accounting for IST/UTC alignments). Uses native Pandas `category` dtypes for categorical features, bypassing the need for manual Label Encoding.

## 📂 Project Architecture

```bash
├── app/
│   ├── app.py             # Main Streamlit Command Center entry point
│   ├── components.py      # Map rendering and UI visualization modules
│   └── heuristics.py      # Rule-based resource allocation engine
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb      # Data ingestion, cleaning, OSM & Weather enrichment
│   └── 02_model_training.ipynb        # Walk-forward splits, XGBoost modeling with OSM features
├── docs/
│   ├── PRESENTATION.md    # 10-slide pitch deck outline
│   └── VIDEO_SCRIPT.md    # Demo video script
├── data/                  # Augmented datasets (OSM + Weather)
├── models/                # Serialized XGBoost models (.json) & Label Encoders
├── requirements.txt       # Core dependencies
├── SUBMISSION.md          # Hackathon submission template
└── README.md              # You are here!
```

## 🚀 Quickstart & Setup

This project uses `uv` as the package manager for lightning-fast dependency resolution.

### 1. Environment Setup

```powershell
# Create a virtual environment using uv
uv venv

# Activate the virtual environment
.venv\Scripts\activate

# Install the required dependencies
uv pip install -r requirements.txt
```

### 2. Running the Data Pipeline (Optional)

The models are already pre-trained and serialized in the `/models` directory. However, if you wish to regenerate the augmented dataset and re-train the models from scratch:

```powershell
uv run jupyter nbconvert --to notebook --execute notebooks/01_eda_and_cleaning.ipynb --inplace
uv run jupyter nbconvert --to notebook --execute notebooks/02_model_training.ipynb --inplace
```

### 3. Launching the Command Center Dashboard

To boot up the live Streamlit dashboard:

```powershell
uv run streamlit run app/app.py
```

_The app will be served locally at `http://localhost:8501`._

## 🧠 Modeling Strategy

We chose an **XGBoost tabular architecture** over deep learning due to the sparsity and tabular nature of the dataset (~7,000 viable rows).

- **Target Construction:** Duration was derived logically by coalescing the `end_datetime` and `closed_datetime`. Outliers (e.g., >48 hours) were clipped to prevent skew.
- **Data Splitting:** A strict time-based walk-forward split was used (80/20 chronological) to completely prevent future data leakage during training.
- **Performance:** Model training takes < 1 second while maintaining high accuracy and low Brier Scores (0.0544), demonstrating extreme computational efficiency ideal for edge-deployment.

## 🛠 Tech Stack

- **Data Processing:** `Pandas`, `NumPy`, `GeoPandas`, `Shapely`, `SciPy`
- **Machine Learning:** `XGBoost`, `Scikit-Learn`, `SHAP`
- **Frontend / Dashboard:** `Streamlit`, `Plotly`, `Folium`, `streamlit-folium`
