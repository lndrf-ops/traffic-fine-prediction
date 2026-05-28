import streamlit as st
import joblib
import pandas as pd

# Import tab modules
from tabs import tab1_explorer, tab2_discovery, tab3_performance, tab4_predictive

# --- 1. CONFIGURATION & DATA LOADING ---
st.set_page_config(page_title="Road Traffic Fines", page_icon="🚦", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('outputs/models/rf_k2.pkl')
    return model

@st.cache_data
def load_data():
    try:
        return pd.read_pickle("data/cleaned/df_cleaned.pkl")
    except FileNotFoundError:
        return None

# Initialization
try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Models not found. Please run 'python run_pipeline.py' first.")
    st.stop()

df_raw = load_data()

# --- 2. HEADER ---
st.title("🚦 Predictive Process Analytics: Road Traffic Fines")
st.markdown("""
This dashboard combines **Process Mining** with **Machine Learning** to understand and predict the outcome of traffic fine cases.
""")

# Erstellung der 4 Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Exploration", 
    "⏳ Process Discovery", 
    "⚖️ Model Performance", 
    "🔮 Predictive System"
])

# --- 3. TABS ---
with tab1:
    tab1_explorer.render(df_raw)

with tab2:
    tab2_discovery.render(df_raw)

with tab3:
    tab3_performance.render()

with tab4:
    tab4_predictive.render(model)