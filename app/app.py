import streamlit as st
import joblib
import json
import pandas as pd

from tabs import (
    tab1_explorer,
    tab2_discovery,
    tab3_performance,
    tab4_predictive,
    tab5_conformance,
    tab6_generative,
)
from theme import apply_theme

import os
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Road Traffic Fines", page_icon="🚦", layout="wide")
apply_theme()

_logo_path = os.path.join(_APP_DIR, "assets", "universityLeipzig.svg")
with open(_logo_path, "r") as f:
    _logo_svg = f.read()
st.markdown(f'<div style="margin-bottom:1rem; max-width:250px;"><img src="data:image/svg+xml;base64,{__import__("base64").b64encode(_logo_svg.encode()).decode()}" style="width:100%;"></div>', unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load XGBoost outcome classifiers for all k/variant combinations."""
    models = {}
    for variant in ["cf", "da"]:
        for k in [2, 3, 5]:
            path = f"outputs/models/xgb_outcome_{variant}_k{k}.pkl"
            try:
                models[(variant, k)] = joblib.load(path)
            except FileNotFoundError:
                pass
    return models


@st.cache_data
def load_data():
    try:
        return pd.read_pickle("data/cleaned/df_cleaned.pkl")
    except FileNotFoundError:
        return None


@st.cache_data
def load_generative_results():
    try:
        with open("outputs/reports/generative_quality_report.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@st.cache_data
def load_synthetic_log():
    try:
        return pd.read_csv("outputs/reports/synthetic_event_log.csv")
    except FileNotFoundError:
        return None


@st.cache_data
def load_completed_cases():
    try:
        return pd.read_pickle("data/cleaned/completed_cases.pkl")
    except FileNotFoundError:
        return None


@st.cache_data
def load_eval_results():
    try:
        with open("outputs/reports/evaluation_results.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@st.cache_data
def load_conformance_results():
    try:
        with open("outputs/reports/conformance_results.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


models = load_models()
if not models:
    st.error("Models not found. Please run 'python run_pipeline.py' first.")
    st.stop()

df_raw = load_data()
generative_results = load_generative_results()
synthetic_log = load_synthetic_log()
completed_cases = load_completed_cases()
eval_results = load_eval_results()
conformance_results = load_conformance_results()

st.title("Road Traffic Fine Management - Process Analytics")
st.caption("Process discovery, conformance checking & predictive modeling on the RTFM event log.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Exploration",
    "Process Discovery",
    "Model Performance",
    "Live Prediction",
    "Conformance",
    "Generative AI",
])

with tab1:
    tab1_explorer.render(df_raw)

with tab2:
    tab2_discovery.render(df_raw)

with tab3:
    tab3_performance.render(eval_results)

with tab4:
    tab4_predictive.render(models)

with tab5:
    tab5_conformance.render(conformance_results)

with tab6:
    tab6_generative.render(generative_results, synthetic_log, completed_cases)
