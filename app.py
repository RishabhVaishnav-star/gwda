import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
DATA = ROOT / "data"
FEATURES = ["rainfall", "temperature", "humidity", "wind_speed",
            "year", "month", "LATITUDE", "LONGITUDE", "lag1", "lag2", "lag3"]
TARGET = "WL_mbgl"

st.set_page_config(
    page_title="GroundSense AI | Water Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#102a43; --muted:#627d98; --teal:#087f8c; --mint:#b8f2e6; --sky:#dff6ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f5fbfd 0%,#f7f9fc 45%,#f3f7fb 100%); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(245,251,253,.88); }
.block-container { padding-top:1.4rem; padding-bottom:3rem; max-width:1450px; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:#102a43; letter-spacing:-.035em; }
h1 { font-weight:800 !important; }
p, label, .stMarkdown { color:#334e68; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#092f3e 0%,#104e5a 55%,#087f8c 100%); }
[data-testid="stSidebar"] * { color:#f0fbff !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:9px 12px; border-radius:12px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(255,255,255,.10); }
.hero { padding:30px 34px; border-radius:26px; color:white; background:linear-gradient(120deg,#073b4c 0%,#087f8c 55%,#42b7a5 100%); box-shadow:0 18px 50px rgba(8,79,91,.16); position:relative; overflow:hidden; margin-bottom:22px; }
.hero:after { content:' '; position:absolute; width:260px; height:260px; right:-55px; top:-90px; border:35px solid rgba(255,255,255,.10); border-radius:50%; }
.hero h1 { color:white !important; font-size:2.5rem !important; line-height:1.08; margin:8px 0 12px; }
.hero p { color:#e5fbff !important; font-size:1.02rem; max-width:760px; }
.eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:800; color:#b8f2e6; }
.hero-pill { display:inline-block; border:1px solid rgba(255,255,255,.35); background:rgba(255,255,255,.13); color:white; padding:6px 11px; border-radius:999px; font-size:.78rem; margin-right:7px; margin-top:7px; }
.section-label { text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:800; color:#087f8c; margin-bottom:5px; }
.subhead { font-family:'Manrope',sans-serif; font-size:1.5rem; font-weight:800; color:#102a43; margin-bottom:4px; }
.muted { color:#627d98; font-size:.93rem; }
.metric-card { background:white; border:1px solid #e2edf2; border-radius:18px; padding:18px 20px; box-shadow:0 8px 24px rgba(16,42,67,.045); min-height:122px; }
.metric-label { color:#627d98; font-size:.8rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
.metric-value { font-family:'Manrope',sans-serif; font-size:1.8rem; font-weight:800; color:#102a43; margin-top:7px; }
.metric-note { color:#829ab1; font-size:.78rem; margin-top:3px; }
.feature-card { background:white; border:1px solid #e2edf2; border-radius:18px; padding:20px; height:100%; box-shadow:0 8px 24px rgba(16,42,67,.035); }
.feature-icon { font-size:1.45rem; }
.feature-title { font-family:'Manrope',sans-serif; font-weight:800; color:#102a43; font-size:1.03rem; margin:8px 0 5px; }
.feature-copy { color:#627d98; font-size:.9rem; line-height:1.55; }
.result-box { border-radius:22px; padding:25px; background:linear-gradient(135deg,#073b4c,#087f8c); color:white; box-shadow:0 15px 35px rgba(8,79,91,.16); }
.result-box * { color:white !important; }
.result-number { font-family:'Manrope',sans-serif; font-size:3rem; font-weight:800; line-height:1.1; margin:9px 0; }
.risk-chip { display:inline-block; border-radius:999px; padding:6px 13px; font-weight:800; font-size:.8rem; }
hr { border-color:#e2edf2; }
div[data-testid="stForm"] { background:white; padding:22px; border:1px solid #e2edf2; border-radius:20px; }
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button { border-radius:12px; border:0; background:linear-gradient(100deg,#087f8c,#20a89a); color:white; font-weight:800; padding:.62rem 1rem; }
.stButton > button:hover, .stFormSubmitButton > button:hover { background:linear-gradient(100deg,#075d69,#168a80); color:white; }
[data-testid="stDataFrame"] { border:1px solid #e2edf2; border-radius:14px; overflow:hidden; }
.small-note { color:#829ab1; font-size:.78rem; }
.callout { background:#e9f9f6; border-left:4px solid #20a89a; border-radius:12px; padding:14px 16px; color:#245b61; }
</style>
""", unsafe_allow_html=True)


def clean_names(s):
    return s.astype(str).str.strip().str.title()


@st.cache_data(show_spinner="Preparing your groundwater intelligence data…")
def prepare_data():
    names = ["janwater_level_data.csv", "augwater_level_data.csv", "pmwater_level_data.csv"]
    pieces = []
    for name in names:
        p = DATA / name
        if not p.exists():
            raise FileNotFoundError(f"Required file is missing: data/{name}")
        d = pd.read_csv(p, low_memory=False)
        d["Season_Source"] = name
        pieces.append(d)
    gw = pd.concat(pieces, ignore_index=True)
    needed = {"STATE_UT", "DISTRICT", "VILLAGE", "LATITUDE", "LONGITUDE", "Date", TARGET}
    absent = needed - set(gw.columns)
    if absent:
        raise ValueError("Groundwater file is missing columns: " + ", ".join(sorted(absent)))
    gw["Date"] = pd.to_datetime(gw["Date"], dayfirst=True, errors="coerce")
    for col in [TARGET, "LATITUDE", "LONGITUDE"]:
        gw[col] = pd.to_numeric(gw[col], errors="coerce")
    gw = gw.dropna(subset=["Date", TARGET, "LATITUDE", "LONGITUDE"])
    gw = gw[gw["STATE_UT"].astype(str).str.contains("Rajasthan", case=False, na=False)].copy()
    gw["DISTRICT"] = clean_names(gw["DISTRICT"])
    if "BLOCK" not in gw:
        gw["BLOCK"] = "Unknown"
    gw["BLOCK"] = clean_names(gw["BLOCK"])
    gw["VILLAGE"] = clean_names(gw["VILLAGE"])
    gw["year"] = gw["Date"].dt.year
    gw["month"] = gw["Date"].dt.month

    cp = DATA / "rajasthan_climate_2000_2025.csv"
    if not cp.exists():
        raise FileNotFoundError("Required file is missing: data/rajasthan_climate_2000_2025.csv")
    climate = pd.read_csv(cp, low_memory=False)
    climate["district"] = clean_names(climate["district"])
    climate["year_month"] = climate["year_month"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    climate["year"] = pd.to_numeric(climate["year_month"].str[:4], errors="coerce")
    climate["month"] = pd.to_numeric(climate["year_month"].str[4:6], errors="coerce")
    for c in ["rainfall", "temperature", "humidity", "wind_speed"]:
        climate[c] = pd.to_numeric(climate[c], errors="coerce")
    climate = climate.dropna(subset=["year", "month"])
    climate["year"] = climate["year"].astype(int)
    climate["month"] = climate["month"].astype(int)

    merged = gw.merge(
        climate[["district", "year", "month", "rainfall", "temperature", "humidity", "wind_speed"]],
        left_on=["DISTRICT", "year", "month"],
        right_on=["district", "year", "month"],
        how="left",
    )
    group = ["DISTRICT", "BLOCK", "VILLAGE", "LATITUDE", "LONGITUDE"]
    merged = merged.sort_values(group + ["Date"]).copy()
    for lag in (1, 2, 3):
        merged[f"lag{lag}"] = merged.groupby(group, dropna=False)[TARGET].shift(lag)
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET]).copy()
    return merged, gw, climate


@st.cache_resource(show_spinner="Training and evaluating the ML model…")
def train(_data):
    cutoff = _data["Date"].quantile(.80)
    train_df = _data[_data["Date"] <= cutoff]
    test_df = _data[_data["Date"] > cutoff]
    if len(train_df) < 100 or len(test_df) < 20:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(_data, test_size=.2, random_state=42)
    model = XGBRegressor(
        n_estimators=250, max_depth=7, learning_rate=.05,
        subsample=.85, colsample_bytree=.9, objective="reg:squarederror",
        random_state=42, n_jobs=-1
    )
    model.fit(train_df[FEATURES], train_df[TARGET])
    pred = model.predict(test_df[FEATURES])
    metrics = {
        "mae": mean_absolute_error(test_df[TARGET], pred),
        "r2": r2_score(test_df[TARGET], pred),
        "train": len(train_df), "test": len(test_df),
        "cutoff": pd.to_datetime(cutoff).strftime("%d %b %Y")
    }
    imp = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    holdout = test_df[["Date", "DISTRICT", TARGET]].copy()
    holdout["Predicted_WL"] = pred
    return model, metrics, imp, holdout


def risk_label(depth):
    if depth > 20:
        return "Critical"
    if depth > 10:
        return "Watch"
    return "Lower depth"


def risk_color(label):
    return {"Critical": "#c2413b", "Watch": "#b7791f", "Lower depth": "#16836c"}.get(label, "#627d98")


def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )


def hero(kicker, title, description, tags=""):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1>'
        f'<p>{description}</p>{tags}</div>',
        unsafe_allow_html=True
    )


try:
    data, gw_raw, climate = prepare_data()
    model, metrics, importance, holdout = train(data)
except Exception as err:
    st.error(f"GroundSense AI couldn't load the project data: {err}")
    st.markdown("Check that the four required source CSVs are in the `data` folder next to `app.py`.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## GroundSense")
    st.markdown("**WATER INTELLIGENCE TOOLKIT**")
    st.markdown("---")
    page = st.radio(
        "WORKSPACE",
        ["Home", "AI Prediction Lab", "District Intelligence", "Explore the Data", "Model Explainability", "About the Project"],
        label_visibility="visible"
    )
    st.markdown("---")
    st.markdown("**LIVE DATA PROFILE**")
    st.caption(f"{len(data):,} model-ready records")
    st.caption(f"{data['DISTRICT'].nunique()} Rajasthan districts")
    st.caption(f"{data['Date'].min():%Y} — {data['Date'].max():%Y}")
    st.markdown("---")
    st.caption("Research prototype • Not an official water authority alert")


# HOME
if page == "Home":
    hero(
        "GROUNDWATER • CLIMATE • MACHINE LEARNING",
        "Understand water before it becomes a crisis.",
        "GroundSense AI transforms groundwater observations and climate signals into understandable risk insights, location-based estimates and practical next steps.",
        '<span class="hero-pill">XGBoost ML</span><span class="hero-pill">Rajasthan focus</span><span class="hero-pill">Climate-aware</span><span class="hero-pill">Interactive analysis</span>'
    )
    c = st.columns(4)
    with c[0]: metric_card("Model-ready observations", f"{len(data):,}", "After feature preparation")
    with c[1]: metric_card("Districts covered", f"{data['DISTRICT'].nunique()}", "In the merged dataset")
    with c[2]: metric_card("Average observed depth", f"{data[TARGET].mean():.1f} m", "Below ground level")
    with c[3]: metric_card("Holdout MAE", f"{metrics['mae']:.2f} m", "Chronological evaluation")

    st.write("")
    left, right = st.columns([1.25, .75], gap="large")
    with left:
        st.markdown('<div class="section-label">THE BIG PICTURE</div><div class="subhead">Groundwater through time</div><div class="muted">Annual average depth across available model-ready observations.</div>', unsafe_allow_html=True)
        yearly = data.groupby(data["Date"].dt.year)[TARGET].mean().reset_index()
        yearly.columns = ["Year", "Average depth"]
        fig = px.area(yearly, x="Year", y="Average depth")
        fig.update_traces(line=dict(width=3), fillcolor="rgba(8,127,140,.16)")
        fig.update_layout(margin=dict(l=8,r=8,t=20,b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Depth (m below ground)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-label">START EXPLORING</div><div class="subhead">Water Intelligence Toolkit</div>', unsafe_allow_html=True)
        for icon, title, body in [
            ("01", "Prediction Lab", "Test climate scenarios for a monitoring location."),
            ("02", "District Intelligence", "Compare groundwater observations across Rajasthan districts."),
            ("03", "Model Explainability", "Understand which model inputs influence estimates."),
            ("04", "Explore the Data", "Filter and examine groundwater and climate records."),
        ]:
            st.markdown(f'<div class="feature-card" style="margin-bottom:10px"><span class="feature-icon">{icon}</span><div class="feature-title">{title}</div><div class="feature-copy">{body}</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-label">READ THIS BEFORE INTERPRETING RESULTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="callout">A larger <b>WL_mbgl</b> value means the water table is deeper below the ground. Risk bands shown here are provisional project thresholds—not official CGWB classifications. R² is not a percentage accuracy score.</div>', unsafe_allow_html=True)


# PREDICTION LAB
elif page == "AI Prediction Lab":
    hero(
        "INTERACTIVE SCENARIO WORKBENCH",
        "AI Prediction Lab",
        "Choose a Rajasthan monitoring location, adjust the climate assumptions, and explore how the model estimates groundwater depth."
    )
    st.markdown('<div class="callout">This is a model-based scenario estimate, not a guaranteed future forecast. Use realistic inputs and treat the output as a screening aid.</div>', unsafe_allow_html=True)
    districts = sorted(data["DISTRICT"].dropna().unique())
    top = st.columns([1,1,1])
    with top[0]:
        district = st.selectbox("1 · Choose district", districts, key="pred_district")
    subset = data[data["DISTRICT"] == district].sort_values("Date")
    sites = subset.groupby(["BLOCK","VILLAGE","LATITUDE","LONGITUDE"], dropna=False).tail(1).copy()
    sites["Site label"] = sites["BLOCK"].astype(str) + " · " + sites["VILLAGE"].astype(str) + " (" + sites["LATITUDE"].round(3).astype(str) + ", " + sites["LONGITUDE"].round(3).astype(str) + ")"
    with top[1]:
        site_label = st.selectbox("2 · Monitoring location", sites["Site label"].tolist())
    row = sites[sites["Site label"] == site_label].iloc[0]
    with top[2]:
        st.markdown(f'<div class="metric-card"><div class="metric-label">LATEST SITE RECORD</div><div class="metric-value" style="font-size:1.25rem">{row["Date"]:%d %b %Y}</div><div class="metric-note">{row["VILLAGE"]}, {district}</div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="subhead">Build a climate scenario</div><div class="muted">Inputs begin with the selected district’s typical values; edit them to test a scenario.</div>', unsafe_allow_html=True)
    med = subset
    with st.form("scenario"):
        col1, col2, col3 = st.columns(3)
        with col1:
            scenario_year = st.number_input("Scenario year", min_value=2000, max_value=2050, value=int(max(2000, min(2025, row["year"]))), step=1)
            scenario_month = st.select_slider("Month", options=list(range(1,13)), value=int(row["month"]), format_func=lambda m: pd.Timestamp(2000,m,1).strftime("%B"))
        with col2:
            rain_med = float(med["rainfall"].median())
            rain = st.number_input("Rainfall · dataset units", min_value=0.0, max_value=2500.0, value=float(max(0, rain_med)), step=5.0)
            temp = st.number_input("Temperature · dataset units", min_value=-10.0, max_value=60.0, value=float(med["temperature"].median()), step=0.5)
        with col3:
            humidity = st.number_input("Humidity · dataset units", min_value=0.0, max_value=100.0, value=float(med["humidity"].median()), step=1.0)
            wind = st.number_input("Wind speed · dataset units", min_value=0.0, max_value=100.0, value=float(max(0, med["wind_speed"].median())), step=0.5)
        run = st.form_submit_button("Generate groundwater estimate", use_container_width=True)
    if run:
        features = pd.DataFrame([{
            "rainfall": rain, "temperature": temp, "humidity": humidity, "wind_speed": wind,
            "year": int(scenario_year), "month": int(scenario_month),
            "LATITUDE": float(row["LATITUDE"]), "LONGITUDE": float(row["LONGITUDE"]),
            "lag1": float(row["lag1"]), "lag2": float(row["lag2"]), "lag3": float(row["lag3"])
        }], columns=FEATURES)
        prediction = float(model.predict(features)[0])
        label = risk_label(prediction)
        st.write("")
        out1, out2 = st.columns([1,1], gap="large")
        with out1:
            st.markdown(f'<div class="result-box"><div class="eyebrow">MODEL ESTIMATE</div><div class="result-number">{prediction:.2f} m</div><div>Estimated depth below ground level</div><br><span class="risk-chip" style="background:{risk_color(label)};color:white">{label.upper()} · PROJECT BAND</span><br><br><div style="color:#d8f7f5;font-size:.88rem">Reference: {district} · {row["VILLAGE"]}<br>Scenario: {pd.Timestamp(int(scenario_year),int(scenario_month),1):%B %Y}</div></div>', unsafe_allow_html=True)
        with out2:
            st.markdown('<div class="subhead">What this result suggests</div>', unsafe_allow_html=True)
            if label == "Critical":
                st.error("The estimate falls in the project's deepest-water-table band. Consider checking historical trends and official local groundwater assessments.")
            elif label == "Watch":
                st.warning("The estimate falls in the project's middle band. Review local trends and seasonal variation before drawing conclusions.")
            else:
                st.success("The estimate falls in the project's lower-depth band. This does not by itself confirm that the location has abundant or sustainable groundwater.")
            st.markdown("**Recent site observations used as model inputs**")
            st.dataframe(pd.DataFrame({"Input": ["Previous reading (lag 1)", "Two readings back (lag 2)", "Three readings back (lag 3)"], "Depth (m)": [row["lag1"],row["lag2"],row["lag3"]]}), hide_index=True, use_container_width=True)
        st.markdown("### Scenario sensitivity")
        st.caption("This chart varies rainfall while keeping the other inputs fixed. It shows model response, not proof that rainfall alone causes the change.")
        rainfall_values = np.linspace(max(0, rain * .25), max(10, rain * 1.75 + 1), 16)
        scenario_rows = []
        for r in rainfall_values:
            f = features.copy()
            f.loc[0, "rainfall"] = r
            scenario_rows.append({"Rainfall": r, "Estimated depth (m)": float(model.predict(f)[0])})
        sens = pd.DataFrame(scenario_rows)
        fig = px.line(sens, x="Rainfall", y="Estimated depth (m)", markers=True)
        fig.add_vline(x=rain, line_dash="dash", annotation_text="Selected input")
        fig.update_layout(margin=dict(l=5,r=5,t=25,b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download this scenario result", pd.DataFrame([{
            "district": district, "site": row["VILLAGE"], "latitude": row["LATITUDE"], "longitude": row["LONGITUDE"],
            "scenario_year": scenario_year, "scenario_month": scenario_month, "rainfall": rain, "temperature": temp,
            "humidity": humidity, "wind_speed": wind, "estimated_depth_mbgl": prediction, "project_risk_band": label
        }]).to_csv(index=False).encode("utf-8"), file_name="groundsense_scenario_result.csv", mime="text/csv")



# DISTRICT INTELLIGENCE
elif page == "District Intelligence":
    hero("REGIONAL COMPARISON", "District Intelligence", "Compare observed groundwater depth across districts, spot areas for further review, and export a regional summary.")
    summary = data.groupby("DISTRICT").agg(
        Observations=(TARGET,"size"), Mean_depth_m=(TARGET,"mean"), Median_depth_m=(TARGET,"median"),
        Deepest_observation_m=(TARGET,"max"), Latest_observation=("Date","max"),
        Latitude=("LATITUDE","mean"), Longitude=("LONGITUDE","mean")
    ).reset_index()
    summary["Project band"] = summary["Mean_depth_m"].apply(risk_label)
    a,b,c = st.columns(3)
    a.metric("Districts represented", len(summary))
    b.metric("Highest district mean", f"{summary['Mean_depth_m'].max():.1f} m")
    b.caption(summary.loc[summary["Mean_depth_m"].idxmax(),"DISTRICT"])
    c.metric("Lowest district mean", f"{summary['Mean_depth_m'].min():.1f} m")
    c.caption(summary.loc[summary["Mean_depth_m"].idxmin(),"DISTRICT"])
    left,right = st.columns([1.1,.9], gap="large")
    with left:
        st.markdown("### Compare district averages")
        fig = px.bar(summary.sort_values("Mean_depth_m",ascending=True).tail(18), x="Mean_depth_m", y="DISTRICT", color="Project band", orientation="h", hover_data=["Observations","Median_depth_m","Deepest_observation_m"], labels={"Mean_depth_m":"Mean depth below ground (m)"})
        fig.update_layout(margin=dict(l=5,r=5,t=15,b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown("### How to read the bands")
        st.markdown('<div class="feature-card"><span class="risk-chip" style="background:#16836c;color:white">LOWER DEPTH</span><p class="feature-copy">Mean observed depth ≤ 10 m. This is not a guarantee of water availability.</p><span class="risk-chip" style="background:#b7791f;color:white">WATCH</span><p class="feature-copy">Mean observed depth > 10 m and ≤ 20 m. Review local trend and seasonal conditions.</p><span class="risk-chip" style="background:#c2413b;color:white">CRITICAL</span><p class="feature-copy">Mean observed depth > 20 m. Prioritize verification against official hydrogeological sources.</p></div>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="small-note">These are simplified project bands applied to average observed depth, not an official groundwater stress index.</div>', unsafe_allow_html=True)
    search = st.text_input("Search districts", placeholder="Type a district name…")
    shown = summary.copy()
    if search:
        shown = shown[shown["DISTRICT"].str.contains(search,case=False,na=False)]
    st.dataframe(shown.sort_values("Mean_depth_m",ascending=False).rename(columns={"Mean_depth_m":"Mean depth (m)","Median_depth_m":"Median depth (m)","Deepest_observation_m":"Deepest observed (m)","Latest_observation":"Latest record"}), use_container_width=True, hide_index=True)
    st.download_button("Export district intelligence CSV", summary.to_csv(index=False).encode("utf-8"), file_name="groundsense_district_intelligence.csv", mime="text/csv")


# DATA EXPLORER
elif page == "Explore the Data":
    hero("DATA DISCOVERY STUDIO", "Explore the evidence", "Filter the underlying records, compare climate and groundwater variables, and inspect the data that feeds the model.")
    x1,x2,x3 = st.columns(3)
    x1.metric("Prepared records", f"{len(data):,}")
    x2.metric("Variables in model", len(FEATURES))
    x3.metric("Date coverage", f"{data['Date'].min():%Y}–{data['Date'].max():%Y}")
    left,right = st.columns([.8,1.2], gap="large")
    with left:
        district_options = ["All districts"] + sorted(data["DISTRICT"].unique())
        chosen = st.selectbox("District", district_options)
        min_year,max_year = int(data["year"].min()),int(data["year"].max())
        years = st.slider("Year range",min_year,max_year,(min_year,max_year))
        variable = st.selectbox("Compare groundwater against", ["rainfall","temperature","humidity","wind_speed"])
    filtered = data[data["year"].between(years[0],years[1])]
    if chosen != "All districts":
        filtered = filtered[filtered["DISTRICT"] == chosen]
    with right:
        fig = px.scatter(filtered.sample(min(5000,len(filtered)),random_state=42), x=variable, y=TARGET, color="DISTRICT", opacity=.62, hover_data=["Date","VILLAGE"], labels={variable:variable.replace("_"," ").title(),TARGET:"Groundwater depth (m below ground)"})
        fig.update_layout(margin=dict(l=5,r=5,t=15,b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Recent matching records")
    cols = ["Date","DISTRICT","BLOCK","VILLAGE","LATITUDE","LONGITUDE",TARGET,"rainfall","temperature","humidity","wind_speed","lag1","lag2","lag3"]
    st.dataframe(filtered[cols].sort_values("Date",ascending=False).head(1000),use_container_width=True,hide_index=True)
    st.download_button("Download filtered data (up to 100,000 rows)", filtered.head(100000).to_csv(index=False).encode("utf-8"), file_name="groundsense_filtered_records.csv", mime="text/csv")


# EXPLAINABILITY
elif page == "Model Explainability":
    hero("MODEL TRANSPARENCY", "Inside the prediction engine", "Understand which features the XGBoost model uses most and how well its predictions match held-out observations.")
    a,b,c = st.columns(3)
    a.metric("Mean absolute error", f"{metrics['mae']:.2f} m", "Average absolute prediction error")
    b.metric("R² score", f"{metrics['r2']:.3f}", "Variance explained on this holdout")
    c.metric("Evaluation records", f"{metrics['test']:,}", f"Training cutoff: {metrics['cutoff']}")
    st.markdown('<div class="callout">These are preliminary metrics from a chronological holdout. R² is not accuracy percentage. Site-wise validation, duplicate checks, and careful handling of seasonal measurements are recommended before making research claims.</div>',unsafe_allow_html=True)
    left,right = st.columns([.9,1.1],gap="large")
    with left:
        st.markdown("### What influences the model?")
        imp = importance.sort_values("Importance",ascending=True)
        fig = px.bar(imp,x="Importance",y="Feature",orientation="h")
        fig.update_layout(margin=dict(l=5,r=5,t=15,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",yaxis_title="")
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Built-in XGBoost feature importance is useful for a first look, but it is not causal evidence or a substitute for SHAP.")
    with right:
        st.markdown("### Observed vs predicted")
        sampled = holdout.sample(min(5000,len(holdout)),random_state=42)
        fig2 = px.scatter(sampled,x=TARGET,y="Predicted_WL",color="DISTRICT",opacity=.65,hover_data=["Date"],labels={TARGET:"Observed depth (m)","Predicted_WL":"Predicted depth (m)"})
        lo = float(min(holdout[TARGET].min(),holdout["Predicted_WL"].min()))
        hi = float(max(holdout[TARGET].max(),holdout["Predicted_WL"].max()))
        fig2.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Ideal match",line=dict(dash="dash",color="#102a43")))
        fig2.update_layout(margin=dict(l=5,r=5,t=15,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2,use_container_width=True)
    st.markdown("### SHAP-based model explanation")
    st.write("SHAP estimates how each model input contributes to a prediction relative to a reference background. Positive values push the estimate toward a deeper water table; negative values push it toward a shallower estimate. These are model explanations, not causal effects.")
    shap_sample_size = st.slider("Number of holdout records to explain", min_value=50, max_value=300, value=100, step=50)
    shap_sample = holdout.sample(min(shap_sample_size, len(holdout)), random_state=42)
    shap_rows = data.loc[shap_sample.index.intersection(data.index), FEATURES] if shap_sample.index.isin(data.index).all() else None
    # Holdout rows are indexed from the prepared dataset; retrieve their exact model inputs.
    if shap_rows is None or len(shap_rows) != len(shap_sample):
        explain_rows = data.loc[shap_sample.index, FEATURES]
    else:
        explain_rows = shap_rows
    show_shap = st.checkbox("Generate SHAP explanations", value=False)
    if show_shap:
        with st.spinner("Calculating SHAP values for the selected records…"):
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(explain_rows)
            shap_matrix = np.asarray(values)
            mean_abs = np.abs(shap_matrix).mean(axis=0)
            shap_importance = pd.DataFrame({"Feature":FEATURES,"Mean absolute SHAP value":mean_abs}).sort_values("Mean absolute SHAP value", ascending=True)
            fig_shap = px.bar(shap_importance, x="Mean absolute SHAP value", y="Feature", orientation="h", title="Global influence across selected records")
            fig_shap.update_layout(margin=dict(l=5,r=5,t=45,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",yaxis_title="")
            st.plotly_chart(fig_shap, use_container_width=True)
            local_idx = st.selectbox("Choose a record for local explanation", list(range(len(explain_rows))), format_func=lambda i: f"Record {i+1} · {shap_sample.iloc[i]['DISTRICT']} · {shap_sample.iloc[i]['Date']:%d %b %Y}")
            exp = shap.Explanation(values=shap_matrix[local_idx], base_values=float(np.asarray(explainer.expected_value).reshape(-1)[0]), data=explain_rows.iloc[local_idx].values, feature_names=FEATURES)
            shap.plots.waterfall(exp, max_display=11, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close("all")
            st.caption("Global SHAP importance is the average absolute contribution magnitude. The local waterfall explains one selected model output against the model's baseline.")
            st.download_button("Download SHAP importance", shap_importance.to_csv(index=False).encode("utf-8"), file_name="groundsense_shap_importance.csv", mime="text/csv")
    else:
        st.info("Enable SHAP explanations when you are ready. The calculation may take a short while on the first run.")
    st.markdown("### Feature ranking")
    st.dataframe(importance.rename(columns={"Feature":"Model input","Importance":"Relative importance"}),use_container_width=True,hide_index=True)
    st.download_button("Download holdout predictions",holdout.to_csv(index=False).encode("utf-8"),file_name="groundsense_holdout_predictions.csv",mime="text/csv")


# ABOUT
elif page == "About the Project":
    hero("THE PROJECT", "GroundSense AI", "An academic machine-learning prototype for groundwater-level estimation, regional exploration and early-warning research.")
    st.markdown("### The problem")
    st.write("Groundwater decisions are difficult when monitoring observations and climate information are spread across different datasets. This project joins groundwater measurements with climate variables and uses historical readings to train a regression model.")
    st.markdown("### Data pipeline")
    steps = st.columns(5)
    for box, n, title, body in zip(steps,["01","02","03","04","05"],["Collect","Prepare","Engineer","Predict","Interpret"],["CGWB observations + NASA POWER climate","Clean dates, locations and missing values","Rainfall, climate, coordinates and lag features","XGBoost regression estimates depth","Charts, district summaries and scenario analysis"]):
        with box:
            st.markdown(f'<div class="feature-card"><div class="section-label">{n}</div><div class="feature-title">{title}</div><div class="feature-copy">{body}</div></div>',unsafe_allow_html=True)
    st.markdown("### Current limitations")
    st.markdown("""
- The website is an academic prototype, not an operational alert system.
- Climate inputs for future scenarios are user-entered assumptions; the model does not itself forecast future weather.
- The displayed project risk bands are simplified and should be checked against official hydrogeological guidance.
- Model performance can be affected by site overlap, seasonal measurement patterns, and repeated observations.
- SHAP values explain model behaviour, not causal groundwater drivers. Interpret them alongside domain knowledge and validation.
""")
    st.markdown("### Next-level improvements")
    st.markdown("""
**Current advanced tool:** global and local SHAP explanations to help interpret model estimates. Future work can add calibrated uncertainty intervals, site-specific hydrological modelling, and verified official-source integrations.
""")
    st.markdown('<div class="small-note">Built for the GroundSense AI B.Tech project • Rajasthan, India</div>',unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#829ab1;font-size:.78rem">GROUNDSENSE AI · WATER INTELLIGENCE TOOLKIT · ACADEMIC PROTOTYPE</div>',unsafe_allow_html=True)
