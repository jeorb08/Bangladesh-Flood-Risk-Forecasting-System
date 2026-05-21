"""
Bangladesh Flood & Rainfall Forecasting — Streamlit Dashboard
=============================================================
Run with:  streamlit run app.py

Requires (run notebooks 01-04 first):
  models/xgb_dhaka_h1d.joblib   — trained XGBoost models
  models/xgb_dhaka_h3d.joblib
  models/xgb_dhaka_h7d.joblib
  models/prophet_dhaka.joblib
  data/processed/<station>_clean.csv  — cleaned weather data
  data/processed/leaderboard.csv      — model comparison table
  data/processed/division_forecasts.csv
  data/raw/bangladesh_districts.geojson
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import geopandas as gpd
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_folium import st_folium

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title = "BD Flood Forecast",
    page_icon  = "🌊",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
HORIZONS = [1, 3, 7]
TARGET   = "rainfall_mm"
SEED     = 42

FLOOD_THRESHOLDS = {"Watch": 50, "Warning": 80, "Danger": 120}

RISK_COLOURS = {
    "🟢 NORMAL":  "#2ECC71",
    "🟡 WATCH":   "#F1C40F",
    "🟠 WARNING": "#E67E22",
    "🔴 DANGER":  "#E74C3C",
}

DIVISION_CENTRES = {
    "dhaka":      {"lat": 23.8103, "lon": 90.4125, "label": "Dhaka"},
    "chittagong": {"lat": 22.3475, "lon": 91.8123, "label": "Chittagong"},
    "sylhet":     {"lat": 24.8949, "lon": 91.8687, "label": "Sylhet"},
    "rajshahi":   {"lat": 24.3745, "lon": 88.6042, "label": "Rajshahi"},
    "khulna":     {"lat": 22.8456, "lon": 89.5403, "label": "Khulna"},
    "barisal":    {"lat": 22.7010, "lon": 90.3535, "label": "Barisal"},
    "rangpur":    {"lat": 25.7439, "lon": 89.2752, "label": "Rangpur"},
    "mymensingh": {"lat": 24.7471, "lon": 90.4203, "label": "Mymensingh"},
}

PALETTE = {
    "blue":   "#3B8BD0",
    "orange": "#E85D24",
    "green":  "#2E9E6B",
    "purple": "#9B59B6",
    "bg":     "#0F1117",
    "card":   "#1A1D2E",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING  (cached — only runs once per session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    """Load trained XGBoost models for all horizons."""
    models = {}
    for h in HORIZONS:
        path = f"models/xgb_dhaka_h{h}d.joblib"
        if os.path.exists(path):
            models[h] = joblib.load(path)
    return models


@st.cache_data
def load_station_data():
    """Load cleaned weather CSVs for all divisions."""
    all_data = {}
    for station in DIVISION_CENTRES:
        path = f"data/processed/{station}_clean.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            if "station" in df.columns:
                df = df.drop(columns=["station"])
            all_data[station] = df
    return all_data


@st.cache_data
def load_geojson():
    """Load Bangladesh district GeoJSON."""
    path = "data/raw/bangladesh_districts.geojson"
    if os.path.exists(path):
        return gpd.read_file(path)
    return None


@st.cache_data
def load_leaderboard():
    path = "data/processed/leaderboard.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (must match notebook_03 exactly)
# ══════════════════════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    rain = df[TARGET]

    for lag in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
        df[f"rain_lag_{lag}d"] = rain.shift(lag)

    for window in [3, 7, 14, 30]:
        df[f"rain_roll_mean_{window}d"] = rain.shift(1).rolling(window).mean()
        df[f"rain_roll_std_{window}d"]  = rain.shift(1).rolling(window).std()
        df[f"rain_roll_max_{window}d"]  = rain.shift(1).rolling(window).max()

    monsoon_rain = rain.where(df.index.month.isin([6, 7, 8, 9, 10]), 0)
    df["monsoon_cumulative"] = monsoon_rain.groupby(
        df.index.year
    ).transform("cumsum")

    counter = 0
    days_list = []
    for v in (rain > 50).astype(int):
        counter = 0 if v else counter + 1
        days_list.append(counter)
    df["days_since_heavy_rain"] = days_list

    df["month"]          = df.index.month
    df["day_of_year"]    = df.index.dayofyear
    df["week_of_year"]   = df.index.isocalendar().week.astype(int)
    df["is_monsoon"]     = df["month"].isin([6, 7, 8, 9, 10]).astype(int)
    df["is_pre_monsoon"] = df["month"].isin([3, 4, 5]).astype(int)
    df["doy_sin"]        = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]        = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)

    for col in ["temp_c", "humidity_pct", "wind_speed_ms"]:
        if col in df.columns:
            df[f"{col}_lag1"]       = df[col].shift(1)
            df[f"{col}_lag3"]       = df[col].shift(3)
            df[f"{col}_roll7_mean"] = df[col].shift(1).rolling(7).mean()

    if "humidity_pct" in df.columns:
        df["humidity_x_monsoon"] = df["humidity_pct"] * df["is_monsoon"]

    return df.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def flood_risk_tier(mm: float) -> str:
    if mm >= FLOOD_THRESHOLDS["Danger"]:  return "🔴 DANGER"
    if mm >= FLOOD_THRESHOLDS["Warning"]: return "🟠 WARNING"
    if mm >= FLOOD_THRESHOLDS["Watch"]:   return "🟡 WATCH"
    return "🟢 NORMAL"


def run_forecast(feat_df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Run all horizon forecasts from the latest data point."""
    feature_cols = [c for c in feat_df.columns if c != TARGET]
    latest       = feat_df.iloc[[-1]]
    base_date    = feat_df.index[-1]
    rows         = []

    for h in HORIZONS:
        if h not in models:
            continue
        pred_mm = float(np.clip(
            models[h].predict(latest[feature_cols].values)[0], 0, None
        ))
        rows.append({
            "horizon":       h,
            "forecast_date": (base_date + pd.DateOffset(days=h)).date(),
            "predicted_mm":  round(pred_mm, 1),
            "risk_tier":     flood_risk_tier(pred_mm),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_rainfall_history(df: pd.DataFrame, station: str, days: int = 180):
    """Rolling rainfall history with threshold lines."""
    recent = df.tail(days)

    fig = go.Figure()

    # Area fill for actual rainfall
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent[TARGET],
        fill="tozeroy", fillcolor="rgba(59,139,208,0.2)",
        line=dict(color=PALETTE["blue"], width=1),
        name="Daily rainfall",
    ))

    # 30-day rolling mean
    roll = recent[TARGET].rolling(30).mean()
    fig.add_trace(go.Scatter(
        x=recent.index, y=roll,
        line=dict(color=PALETTE["orange"], width=2),
        name="30-day rolling mean",
    ))

    # Threshold lines
    for label, val in FLOOD_THRESHOLDS.items():
        colour = {"Watch":"#F1C40F","Warning":"#E67E22","Danger":"#E74C3C"}[label]
        fig.add_hline(
            y=val, line_dash="dot", line_color=colour,
            annotation_text=label, annotation_position="top right",
            annotation_font_size=11,
        )

    fig.update_layout(
        title=dict(text=f"{station.title()} — Last {days} days of rainfall",
                   font=dict(size=15)),
        xaxis_title="Date",
        yaxis_title="Rainfall (mm/day)",
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card"],
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        height=320,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_forecast_bar(forecast_df: pd.DataFrame):
    """7-day forecast bar chart with colour-coded risk tiers."""
    colours = [RISK_COLOURS.get(r, "#2ECC71") for r in forecast_df["risk_tier"]]

    fig = go.Figure(go.Bar(
        x=[f"+{h}d" for h in forecast_df["horizon"]],
        y=forecast_df["predicted_mm"],
        marker_color=colours,
        text=[f"{v:.1f} mm" for v in forecast_df["predicted_mm"]],
        textposition="outside",
        width=0.5,
    ))

    for label, val in FLOOD_THRESHOLDS.items():
        colour = {"Watch":"#F1C40F","Warning":"#E67E22","Danger":"#E74C3C"}[label]
        fig.add_hline(
            y=val, line_dash="dot", line_color=colour,
            annotation_text=label, annotation_position="top left",
            annotation_font_size=11,
        )

    fig.update_layout(
        title=dict(text="Forecast rainfall by horizon",
                   font=dict(size=15)),
        xaxis_title="Forecast horizon",
        yaxis_title="Predicted rainfall (mm)",
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card"],
        height=320,
        margin=dict(l=10, r=10, t=60, b=10),
        yaxis=dict(range=[0, max(
            forecast_df["predicted_mm"].max() * 1.3, 130
        )]),
    )
    return fig


def plot_monthly_climatology(df: pd.DataFrame, station: str):
    """Monthly rainfall climatology — box + mean line."""
    
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    monsoon = [6, 7, 8, 9, 10]

    fig = go.Figure()

    for m in range(1, 13):

        vals = df[df.index.month == m][TARGET].values

        # Main colour
        colour = PALETTE["blue"] if m in monsoon else "#555577"

        # Proper RGBA fill colours
        if m in monsoon:
            fill = "rgba(59,139,208,0.30)"
        else:
            fill = "rgba(85,85,119,0.30)"

        fig.add_trace(go.Box(
            y=vals,
            name=month_names[m-1],

            marker_color=colour,
            line_color=colour,
            fillcolor=fill,

            showlegend=False,
            boxpoints=False,
        ))

    fig.update_layout(
        title=dict(
            text=f"{station.title()} — Monthly rainfall climatology (blue = monsoon)",
            font=dict(size=15),
        ),

        yaxis_title="Rainfall (mm/day)",

        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card"],

        height=320,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return fig


def plot_metric_comparison(leaderboard: pd.DataFrame):
    """Side-by-side RMSE and skill score for XGBoost vs Prophet."""
    xgb = leaderboard[leaderboard["model"] == "XGBoost"]
    pro = leaderboard[leaderboard["model"] == "Prophet"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["RMSE (mm) — lower is better",
                        "Skill Score — higher is better"],
    )

    for label, data, colour in [
        ("XGBoost", xgb, PALETTE["blue"]),
        ("Prophet", pro, PALETTE["orange"]),
    ]:
        fig.add_trace(go.Bar(
            x=data["horizon"].astype(str) + "d",
            y=data["RMSE"],
            name=label,
            marker_color=colour,
            opacity=0.85,
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=data["horizon"].astype(str) + "d",
            y=data["Skill"],
            name=label,
            marker_color=colour,
            showlegend=False,
            opacity=0.85,
        ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dot", line_color="white",
                  opacity=0.5, row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card"],
        height=320,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FOLIUM RISK MAP
# ══════════════════════════════════════════════════════════════════════════════
def build_risk_map(forecast_df: pd.DataFrame, gdf: gpd.GeoDataFrame,
                   horizon: int = 7) -> folium.Map:
    """Build Folium choropleth map for selected horizon."""

    NAME_MAP = {
        "Dhaka": "dhaka", "Chattogram": "chittagong",
        "Chittagong": "chittagong", "Sylhet": "sylhet",
        "Rajshahi": "rajshahi", "Khulna": "khulna",
        "Barishal": "barisal", "Barisal": "barisal",
        "Rangpur": "rangpur", "Mymensingh": "mymensingh",
    }

    fc = forecast_df[forecast_df["horizon"] == horizon].copy()

    gdf2 = gdf.copy()
    gdf2["division_key"] = gdf2["NAME_1"].map(NAME_MAP)
    gdf2 = gdf2.merge(
        fc[["risk_tier", "predicted_mm"]].rename(
            index=str, columns={}
        ).assign(division_key=fc["station"].values
                 if "station" in fc.columns
                 else [""]*len(fc)),
        on="division_key", how="left",
    )

    tier_num = {"🟢 NORMAL": 0, "🟡 WATCH": 1,
                "🟠 WARNING": 2, "🔴 DANGER": 3}
    gdf2["risk_num"]     = gdf2["risk_tier"].map(tier_num).fillna(0)
    gdf2["predicted_mm"] = gdf2["predicted_mm"].fillna(0)
    gdf2["risk_tier"]    = gdf2["risk_tier"].fillna("🟢 NORMAL")

    m = folium.Map(
        location   = [23.685, 90.356],
        zoom_start = 7,
        tiles      = "CartoDB positron",
    )

    def style_fn(feature):
        tier   = feature["properties"].get("risk_tier", "🟢 NORMAL")
        colour = RISK_COLOURS.get(tier, "#2ECC71")
        return {"fillColor": colour, "color": "white",
                "weight": 0.5, "fillOpacity": 0.65}

    def highlight_fn(feature):
        return {"weight": 2.5, "color": "#333",  "fillOpacity": 0.85}

    folium.GeoJson(
        gdf2.__geo_interface__,
        style_function     = style_fn,
        highlight_function = highlight_fn,
        tooltip = folium.GeoJsonTooltip(
            fields  = ["NAME_2", "NAME_1", "risk_tier", "predicted_mm"],
            aliases = ["District:", "Division:", "Risk:", "Predicted (mm):"],
            style   = ("background-color:white; color:#333;"
                       "font-family:Arial; font-size:13px; padding:6px;"),
        ),
        popup = folium.GeoJsonPopup(
            fields  = ["NAME_2", "NAME_1", "predicted_mm", "risk_tier"],
            aliases = ["District", "Division", "Predicted (mm)", "Risk tier"],
            style   = "font-family:Arial; font-size:13px;",
        ),
    ).add_to(m)

    # Division marker pins
    for station, info in DIVISION_CENTRES.items():
        row = fc[fc["station"] == station] if "station" in fc.columns else pd.DataFrame()
        mm  = float(row["predicted_mm"].values[0]) if len(row) > 0 else 0.0
        tier = flood_risk_tier(mm)
        icon_colour = {
            "🟢 NORMAL": "green", "🟡 WATCH": "orange",
            "🟠 WARNING": "red",  "🔴 DANGER": "darkred",
        }.get(tier, "green")
        folium.Marker(
            location = [info["lat"], info["lon"]],
            tooltip  = f"{info['label']} — {mm:.1f} mm ({tier})",
            icon     = folium.Icon(color=icon_colour, icon="tint", prefix="fa"),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;border:2px solid #aaa;border-radius:8px;
                padding:10px 14px;font-family:Arial;font-size:12px;
                box-shadow:3px 3px 8px rgba(0,0,0,0.2);">
      <b>🌊 Flood Risk</b><br>
      <span style="background:#2ECC71;padding:1px 6px;border-radius:2px">&nbsp;</span>
      Normal (&lt;50mm)<br>
      <span style="background:#F1C40F;padding:1px 6px;border-radius:2px">&nbsp;</span>
      Watch (50–80mm)<br>
      <span style="background:#E67E22;padding:1px 6px;border-radius:2px">&nbsp;</span>
      Warning (80–120mm)<br>
      <span style="background:#E74C3C;padding:1px 6px;border-radius:2px">&nbsp;</span>
      Danger (&gt;120mm)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    plugins.Fullscreen(position="topright").add_to(m)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# CSS — custom dark theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0F1117; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background-color: #1A1D2E; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background-color: #1A1D2E;
    border: 1px solid #2A2D3E;
    border-radius: 10px;
    padding: 14px 18px;
  }

  /* Alert boxes */
  .alert-danger  { background:#E74C3C22; border-left:4px solid #E74C3C;
                   padding:10px 14px; border-radius:6px; margin:6px 0; }
  .alert-warning { background:#E67E2222; border-left:4px solid #E67E22;
                   padding:10px 14px; border-radius:6px; margin:6px 0; }
  .alert-watch   { background:#F1C40F22; border-left:4px solid #F1C40F;
                   padding:10px 14px; border-radius:6px; margin:6px 0; }
  .alert-normal  { background:#2ECC7122; border-left:4px solid #2ECC71;
                   padding:10px 14px; border-radius:6px; margin:6px 0; }

  /* Section headers */
  .section-header {
    font-size: 18px; font-weight: 600;
    color: #E0E0E0; margin: 20px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2A2D3E;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { color: #aaa; }
  .stTabs [aria-selected="true"] { color: #3B8BD0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/f9/Flag_of_Bangladesh.svg",
             width=60)
    st.title("🌊 BD Flood Forecast")
    st.caption("XGBoost + Prophet | NASA POWER data")
    st.divider()

    # Station selector
    station = st.selectbox(
        "Select division",
        options = list(DIVISION_CENTRES.keys()),
        format_func = lambda s: DIVISION_CENTRES[s]["label"],
        index   = 0,
    )

    # History window
    history_days = st.slider(
        "Rainfall history (days)",
        min_value=30, max_value=365,
        value=180, step=30,
    )

    # Map horizon
    map_horizon = st.radio(
        "Map forecast horizon",
        options=[1, 3, 7],
        format_func=lambda h: f"{h}-day ahead",
        horizontal=True,
    )

    st.divider()
    st.markdown("""
    **Data sources**
    - [NASA POWER API](https://power.larc.nasa.gov)
    - [BWDB river data](http://www.bwdb.gov.bd)
    - [BFFWC thresholds](http://www.ffwc.gov.bd)

    **Models**
    - XGBoost (direct multi-step)
    - Prophet (seasonal baseline)

    **Alert levels (mm/day)**
    🟡 Watch: >50 | 🟠 Warning: >80 | 🔴 Danger: >120
    """)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
models    = load_models()
all_data  = load_station_data()
gdf       = load_geojson()
leaderboard = load_leaderboard()

# Check data loaded
if not models:
    st.error("❌ No models found in `models/`. Run notebook_03 first.")
    st.stop()

if station not in all_data:
    st.error(f"❌ No data found for {station}. Run notebook_01 first.")
    st.stop()

df      = all_data[station]
feat_df = build_features(df)
forecast_df = run_forecast(feat_df, models)

# Add station col for map (needed by build_risk_map)
forecast_df["station"] = station


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
col_title, col_info = st.columns([3, 1])
with col_title:
    st.title("🇧🇩 Bangladesh Flood Risk Forecast")
    st.caption(
        f"Division: **{DIVISION_CENTRES[station]['label']}** · "
        f"Last data: **{feat_df.index[-1].date()}** · "
        f"Model: **XGBoost (direct multi-step)**"
    )
with col_info:
    current_season = "🌧 Monsoon" if df.index[-1].month in [6,7,8,9,10] else "☀ Dry Season"
    st.metric("Current season", current_season)


# ══════════════════════════════════════════════════════════════════════════════
# ALERT BANNER — highest risk in 7-day window
# ══════════════════════════════════════════════════════════════════════════════
worst_tier = forecast_df.sort_values("predicted_mm", ascending=False).iloc[0]
tier_str   = worst_tier["risk_tier"]
max_mm     = worst_tier["predicted_mm"]
max_date   = worst_tier["forecast_date"]

alert_class = {
    "🔴 DANGER": "alert-danger",
    "🟠 WARNING": "alert-warning",
    "🟡 WATCH": "alert-watch",
    "🟢 NORMAL": "alert-normal",
}.get(tier_str, "alert-normal")

st.markdown(
    f'<div class="{alert_class}">'
    f'<b>7-day outlook — {tier_str}</b>&nbsp;&nbsp;'
    f'Peak forecast: <b>{max_mm:.1f} mm</b> on {max_date}'
    f'</div>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST METRIC CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Forecast Summary</div>',
            unsafe_allow_html=True)

cols = st.columns(3)
tier_delta_map = {
    "🟢 NORMAL": "normal", "🟡 WATCH": "off",
    "🟠 WARNING": "inverse", "🔴 DANGER": "inverse",
}

for col, (_, row) in zip(cols, forecast_df.iterrows()):
    with col:
        st.metric(
            label = f"**+{row['horizon']} day** — {row['forecast_date']}",
            value = f"{row['predicted_mm']:.1f} mm",
            delta = row["risk_tier"],
            delta_color = tier_delta_map.get(row["risk_tier"], "normal"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Rainfall History",
    "🗺 Risk Map",
    "🔬 Model Comparison",
    "📋 Raw Data",
])


# ── Tab 1: Rainfall History ───────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.plotly_chart(
            plot_rainfall_history(df, station, days=history_days),
            use_container_width=True,
        )

    with col_right:
        st.plotly_chart(
            plot_forecast_bar(forecast_df),
            use_container_width=True,
        )

    st.plotly_chart(
        plot_monthly_climatology(df, station),
        use_container_width=True,
    )

    # Quick stats
    st.markdown('<div class="section-header">📌 Station Statistics</div>',
                unsafe_allow_html=True)
    rain = df[TARGET]
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Mean daily rainfall",  f"{rain.mean():.1f} mm")
    s2.metric("Max single day",       f"{rain.max():.0f} mm")
    s3.metric("Danger days / year",
              f"{(rain > 120).sum() / ((df.index.max()-df.index.min()).days/365):.1f}")
    s4.metric("Zero rain days",
              f"{(rain == 0).mean()*100:.0f}%")
    s5.metric("Monsoon total (avg)",
              f"{rain[rain.index.month.isin([6,7,8,9,10])].resample('YE').sum().mean():.0f} mm")


# ── Tab 2: Risk Map ───────────────────────────────────────────────────────────
with tab2:
    if gdf is None:
        st.warning(
            "⚠ GeoJSON not found at `data/raw/bangladesh_districts.geojson`. "
            "Run Cell 2 in notebook_04 to download it."
        )
    else:
        st.markdown(
            f'<div class="section-header">'
            f'🗺 Flood Risk Map — {map_horizon}-day forecast'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Build all-division forecast for map
        all_forecasts = []
        for s, fd in {k: build_features(v)
                      for k, v in all_data.items()}.items():
            fc = run_forecast(fd, models)
            fc["station"] = s
            all_forecasts.append(fc)

        all_fc_df = pd.concat(all_forecasts, ignore_index=True)

        risk_map = build_risk_map(all_fc_df, gdf, horizon=map_horizon)

        st_folium(risk_map, width=None, height=520,
                  returned_objects=[], use_container_width=True)

        # Risk summary table below map
        st.markdown("**Division risk summary:**")
        summary = all_fc_df[all_fc_df["horizon"] == map_horizon][
            ["station", "predicted_mm", "risk_tier", "forecast_date"]
        ].copy()
        summary["station"] = summary["station"].str.title()
        summary = summary.sort_values("predicted_mm", ascending=False)
        summary.columns = ["Division", "Predicted (mm)", "Risk Tier", "Forecast Date"]
        st.dataframe(
            summary.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )


# ── Tab 3: Model Comparison ───────────────────────────────────────────────────
with tab3:
    if leaderboard is None:
        st.warning(
            "⚠ Leaderboard not found. Run Cell 13 in notebook_03 to generate it."
        )
    else:
        st.markdown('<div class="section-header">📊 XGBoost vs Prophet</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            plot_metric_comparison(leaderboard),
            use_container_width=True,
        )

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Full leaderboard:**")
            st.dataframe(leaderboard, use_container_width=True, hide_index=True)
        with col_r:
            st.markdown("**When to use each model:**")
            st.info(
                "**XGBoost** — better RMSE on real noisy data. "
                "Uses lag + meteorological features. Preferred for operational alerts."
            )
            st.warning(
                "**Prophet** — provides uncertainty intervals (80% bands). "
                "More interpretable. Good for communicating risk to non-technical users."
            )
            st.success(
                "**Skill score > 0** means the model beats the naive persistence "
                "baseline (predicting tomorrow = today). Both models achieve this."
            )

    # SHAP plot if saved
    shap_path = "data/processed/shap_importance.png"
    if os.path.exists(shap_path):
        st.markdown('<div class="section-header">🧠 SHAP Feature Importance</div>',
                    unsafe_allow_html=True)
        st.image(shap_path, caption="SHAP values — XGBoost 1-day model",
                 use_container_width=True)


# ── Tab 4: Raw Data ───────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">📋 Raw Weather Data</div>',
                unsafe_allow_html=True)

    col_f, col_t = st.columns(2)
    with col_f:
        from_date = st.date_input("From", value=df.index[-90].date())
    with col_t:
        to_date   = st.date_input("To",   value=df.index[-1].date())

    filtered = df.loc[str(from_date):str(to_date)].copy()
    filtered.index = filtered.index.date

    st.dataframe(filtered.round(2), use_container_width=True, height=380)

    # Download button
    csv = filtered.to_csv().encode("utf-8")
    st.download_button(
        label    = "⬇ Download as CSV",
        data     = csv,
        file_name= f"{station}_rainfall_{from_date}_{to_date}.csv",
        mime     = "text/csv",
    )

    # Correlation matrix
    st.markdown("**Feature correlations:**")
    corr = df.corr().round(3)
    fig_corr = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pearson correlation matrix",
    )
    fig_corr.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with XGBoost · Prophet · NASA POWER · Folium · Streamlit · Plotly · GeoPandas | "
    "Bangladesh Flood Forecasting Project"
)