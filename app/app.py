import streamlit as st
import joblib
import pandas as pd

# Importiere unsere ausgelagerten Tabs
from tabs import tab1_explorer, tab2_discovery, tab3_performance, tab4_predictive

# --- 1. KONFIGURATION & DATEN LADEN ---
st.set_page_config(page_title="Road Traffic Fines", page_icon="🚦", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('models/rf_model.pkl')
    features = joblib.load('models/model_features.pkl')
    return model, features

@st.cache_data
def load_data():
    try:
        return pd.read_pickle("data/processed/df_raw.pkl")
    except FileNotFoundError:
        return None

# Initialisierung
try:
    model, features = load_model()
except FileNotFoundError:
    st.error("⚠️ Modelle nicht gefunden. Bitte 'python run_pipeline.py' ausführen.")
    st.stop()

df_raw = load_data()

# --- 2. HEADER ---
st.title("🚦 Predictive Process Analytics: Road Traffic Fines")
st.markdown("""
Dieses Dashboard kombiniert **Process Mining** mit **Machine Learning**, um den Ausgang von Bußgeldverfahren zu verstehen und vorherzusagen.
""")

# Erstellung der 4 Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Explorer", 
    "⏳ Process Discovery", 
    "⚖️ Model Performance", 
    "🔮 Predictive System"
])

# --- 3. TABS AUFRUFEN ---
with tab1:
    tab1_explorer.render(df_raw)

with tab2:
    tab2_discovery.render(df_raw)

with tab3:
    tab3_performance.render()

with tab4:
    tab4_predictive.render(model, features)