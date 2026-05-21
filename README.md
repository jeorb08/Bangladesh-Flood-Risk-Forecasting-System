
# 🌊 Bangladesh Flood & Rainfall Forecasting System

> Multi-step rainfall forecasting for all 8 Bangladesh divisions using XGBoost and Prophet — with an interactive district-level flood risk map and live Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange?logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?logo=streamlit)
![NASA POWER](https://img.shields.io/badge/Data-NASA%20POWER-darkblue?logo=nasa)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Bangladesh is one of the most flood-prone countries on Earth — with over 80% of its landmass lying on floodplains and roughly **20 million people displaced by floods every year**. Despite this, open-source forecasting tools for Bangladeshi rainfall remain rare.

This project builds an **end-to-end flood risk forecasting pipeline** that:

- Fetches **23 years** of daily weather data from the NASA POWER API (no cost, no account)
- Engineers **35+ lag, rolling, and calendar features** to capture monsoon dynamics
- Trains **XGBoost models** for 1-day, 3-day, and 7-day rainfall forecasts
- Benchmarks against a **Prophet seasonal baseline**
- Classifies each division into **BFFWC flood alert tiers** (Normal / Watch / Warning / Danger)
- Renders an **interactive Folium choropleth map** across all 64 districts
- Wraps everything in a **Streamlit dashboard** with live forecasts and a downloadable data table

---

## 🖥 Live Demo

<img width="1918" height="906" alt="Screenshot 2026-05-21 201633" src="https://github.com/user-attachments/assets/bd361770-8c97-41f1-95f2-c3f25ca7e636" />

<img width="1655" height="613" alt="Screenshot 2026-05-21 201804" src="https://github.com/user-attachments/assets/619b8677-2eda-4aa9-a8f4-73d628879668" />

<img width="1661" height="727" alt="Screenshot 2026-05-21 201833" src="https://github.com/user-attachments/assets/259b57d7-dda8-4721-b5f2-ee554c98246c" />

<img width="1637" height="491" alt="Screenshot 2026-05-21 201901" src="https://github.com/user-attachments/assets/7dc9fa9a-886a-4f51-8ba2-e50fd7315a6f" />

<img width="1566" height="847" alt="Screenshot 2026-05-21 201929" src="https://github.com/user-attachments/assets/3d66be5e-5671-4a86-8d21-40778f05ca56" />





---

## 📂 Project Structure

```
bangladesh_flood/
│
├── 📓 Notebooks (run in order)
│   ├── notebook_01_data.ipynb          # Fetch NASA POWER data, save CSVs
│   ├── notebook_02_eda.ipynb           # EDA: distributions, ACF/PACF, decomposition
│   ├── notebook_03_features_model.ipynb  # Features, XGBoost, Prophet, SHAP
│   └── notebook_04_map.ipynb           # Folium risk map + HTML report
│
├── 🐍 Python helpers
│   └── nasa_power_fetcher.py           # NASA POWER API client + feature builder
│
├── 🚀 Dashboard
│   └── app.py                          # Streamlit app (all tabs, map, charts)
│
├── 📁 data/
│   ├── raw/                            # NASA POWER CSVs, GeoJSON
│   └── processed/                      # Cleaned data, leaderboard, map HTMLs
│
├── 📁 models/
│   ├── xgb_dhaka_h1d.joblib            # XGBoost — 1-day forecast
│   ├── xgb_dhaka_h3d.joblib            # XGBoost — 3-day forecast
│   ├── xgb_dhaka_h7d.joblib            # XGBoost — 7-day forecast
│   └── prophet_dhaka.joblib            # Prophet baseline model
│
└── requirements.txt
```

---

## 📊 Model Results

### XGBoost vs Prophet — Dhaka Division

| Horizon | XGBoost RMSE | Prophet RMSE | XGBoost Skill Score |
|---------|-------------|--------------|---------------------|
| 1-day   | **23.43 mm**| 23.13 mm     | +0.293              |
| 3-day   | **23.60 mm**| 23.16 mm     | +0.289              |
| 7-day   | **23.75 mm**| 23.23 mm     | +0.286              |

> **Skill score > 0** means the model outperforms the naive persistence baseline (predicting tomorrow = today).
>
> **Note on MAPE:** Rainfall MAPE appears inflated (~230%) due to near-zero dry-season values. RMSE and skill score are the correct evaluation metrics for this domain.

### Why both models?

| Model | Strength | Best used for |
|-------|----------|---------------|
| **XGBoost** | Lower RMSE on real noisy data, uses meteorological lag features | Operational alert generation |
| **Prophet** | 80% uncertainty intervals, interpretable decomposition | Communicating risk to non-technical users |

---

## 🌍 Flood Alert Tiers

Based on **Bangladesh Flood Forecasting & Warning Centre (BFFWC)** official thresholds:

| Tier | Threshold | Colour |
|------|-----------|--------|
| 🟢 Normal | < 50 mm/day | Green |
| 🟡 Watch | 50 – 80 mm/day | Yellow |
| 🟠 Warning | 80 – 120 mm/day | Orange |
| 🔴 Danger | > 120 mm/day | Red |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/bangladesh-flood-forecast.git
cd bangladesh-flood-forecast
```

### 2. Create virtual environment

```bash
python -m venv flood_env

# Windows
flood_env\Scripts\activate

# macOS / Linux
source flood_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install streamlit-folium   # extra package for Folium inside Streamlit
```

---

## 🚀 How to Run

Run the notebooks in order — each builds on the previous one.

```bash
# Open Jupyter
jupyter notebook
```

| Step | Notebook | What it does | Time |
|------|----------|--------------|------|
| 1 | `notebook_01_data.ipynb` | Fetch 23yr NASA POWER data for 8 divisions | ~3 min |
| 2 | `notebook_02_eda.ipynb` | EDA, ADF test, ACF/PACF, decomposition | ~2 min |
| 3 | `notebook_03_features_model.ipynb` | Build features, train XGBoost + Prophet | ~10 min |
| 4 | `notebook_04_map.ipynb` | Generate Folium risk map + HTML report | ~2 min |

Then launch the dashboard:

```bash
streamlit run app.py
```

---

## 🔬 Technical Deep Dive

### Feature Engineering (35+ features)

```
Lag features       : rain_lag_1d, rain_lag_3d, rain_lag_7d, rain_lag_14d ...
Rolling statistics : rain_roll_mean_7d, rain_roll_std_30d, rain_roll_max_14d ...
Calendar features  : doy_sin, doy_cos, month_sin, month_cos, is_monsoon ...
Meteorological     : temp_c_lag1, humidity_pct_roll7_mean, wind_speed_ms_lag3 ...
Interaction        : humidity_x_monsoon (key for monsoon onset detection)
Memory features    : days_since_heavy_rain, monsoon_cumulative
```

### Multi-step Forecasting Strategy

Uses the **direct strategy** — one independent XGBoost model per horizon:

```
Features at time t  →  Model H=1  →  Prediction at t+1
Features at time t  →  Model H=3  →  Prediction at t+3
Features at time t  →  Model H=7  →  Prediction at t+7
```

This avoids **error accumulation** from the recursive strategy (where errors compound across each step).

### Hyperparameter Tuning

```python
# Optuna TPE sampler — 40 trials per horizon
optuna.create_study(direction="minimize")
# Search space: n_estimators, max_depth, learning_rate,
#               subsample, colsample_bytree, reg_alpha, reg_lambda
```

### Experiment Tracking

All runs logged to **MLflow**. Launch the UI with:

```bash
mlflow ui --port 5000
# Then open: http://localhost:5000
```

---

## 🗺 Interactive Risk Map

The Folium map shows all **64 Bangladesh districts** coloured by predicted flood risk tier, with:
- Clickable district popups (district name, division, predicted mm, risk tier)
- Division marker pins with rainfall forecast
- Toggle between 1-day, 3-day, 7-day horizons

Open `data/processed/flood_dashboard.html` directly in any browser — no server needed.

---

## 📡 Data Sources

| Source | Data | Access |
|--------|------|--------|
| [NASA POWER API](https://power.larc.nasa.gov) | Daily rainfall, temperature, humidity, wind speed (1981–present) | Free, no account |
| [BWDB](http://www.bwdb.gov.bd) | River water levels and discharge | Public portal |
| [GADM](https://gadm.org) | Bangladesh district boundary GeoJSON | Free download |
| [BFFWC](http://www.ffwc.gov.bd) | Official flood alert thresholds | Public reference |

---

## 🚢 Deployment

### Streamlit Community Cloud (free)

1. Push this repo to GitHub (make sure `requirements.txt` is included)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → set main file as `app.py`
5. Click **Deploy**

> Pre-train models locally and commit the `models/` folder to GitHub, or add a `@st.cache_resource` model-training step inside `app.py` for cloud retraining.

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| **Data** | pandas, numpy, requests (NASA POWER API) |
| **Modelling** | XGBoost, Prophet, scikit-learn |
| **Tuning** | Optuna (TPE sampler, 40 trials/horizon) |
| **Explainability** | SHAP (TreeExplainer, beeswarm + bar plots) |
| **Tracking** | MLflow (params, metrics, model artifacts) |
| **Geospatial** | GeoPandas, Folium, Shapely |
| **Dashboard** | Streamlit, Plotly, streamlit-folium |
| **Persistence** | joblib (.joblib model files) |

---

## 📈 Skills Demonstrated

- Time series forecasting (multi-step, direct strategy)
- Feature engineering (lag, rolling, cyclical encoding)
- Hyperparameter tuning with Optuna
- Model explainability with SHAP
- Geospatial data processing and choropleth mapping
- MLflow experiment tracking
- End-to-end ML pipeline from data fetch to deployed dashboard
- Real-world impact framing (Bangladesh flood risk)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [NASA POWER](https://power.larc.nasa.gov) for providing free global weather data
- [BFFWC Bangladesh](http://www.ffwc.gov.bd) for official flood threshold definitions
- [GADM](https://gadm.org) for Bangladesh district boundary data

---

