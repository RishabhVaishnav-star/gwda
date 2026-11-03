# GroundSense AI — Water Intelligence Toolkit

A polished Streamlit application for groundwater scenario estimation, district-level analysis, dataset exploration, and SHAP model interpretation.

## Run locally (Windows)
1. Extract the ZIP.
2. Open Command Prompt inside the `GroundSenseAI_Website` folder.
3. Run:
   ```bat
   py -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Open the local address printed by Streamlit (usually `http://localhost:8501`).

The first launch prepares the source data and trains XGBoost, so it can take a few minutes. SHAP calculations are optional and are triggered from the Model Explainability page.

## Main tools
- **AI Prediction Lab:** enter a district, monitoring location and climate assumptions to generate a groundwater-depth scenario estimate.
- **District Intelligence:** compare district-level observed groundwater statistics and project screening bands.
- **Explore the Data:** filter and inspect the prepared groundwater-climate dataset.
- **Model Explainability:** review evaluation metrics, XGBoost feature importance, observed-versus-predicted values and global/local SHAP explanations.

## Required CSVs inside `data/`
- `janwater_level_data.csv`
- `augwater_level_data.csv`
- `pmwater_level_data.csv`
- `rajasthan_climate_2000_2025.csv`

The previous `groundsense_dashboard.csv` is included as a project artifact; the redesigned app recalculates summaries from the source data.

## Research and interpretation notes
- `WL_mbgl` is depth below ground level. Larger values mean a deeper water table.
- The displayed bands (≤10 m, >10–20 m, >20 m) are provisional project thresholds, not universal official CGWB categories.
- R² is not percentage accuracy. MAE is reported in the target's units.
- The model uses a chronological holdout, but site-wise validation and duplicate / seasonal-observation checks are still needed.
- Scenario estimates are not guaranteed future forecasts. Climate assumptions are entered by the user.
- SHAP describes how model inputs contribute to model outputs; it does not establish causation.

## Project stack
Python, Streamlit, Pandas, NumPy, Plotly, XGBoost, scikit-learn, SHAP and Matplotlib.

