import streamlit as st
import joblib
import json
import pandas as pd

from tabs import tab1_explorer, tab2_discovery, tab3_performance, tab4_predictive, tab5_conformance

st.set_page_config(page_title="Road Traffic Fines", page_icon="🚦", layout="wide")


@st.cache_resource
def load_models():
    """Load XGBoost outcome classifiers for all k/variant combinations."""
    models = {}
    for variant in ["cf", "da"]:
        for k in [2, 3, 5, 8]:
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
eval_results = load_eval_results()
conformance_results = load_conformance_results()

st.title("Predictive Process Analytics: Road Traffic Fines")
st.markdown(
    "This dashboard combines **Process Mining** with **Machine Learning** "
    "to understand and predict the outcome of traffic fine cases."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Exploration",
    "Process Discovery",
    "Model Performance",
    "Live Prediction",
    "Conformance",
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
