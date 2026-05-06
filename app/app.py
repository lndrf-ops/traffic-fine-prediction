import streamlit as st
import joblib
import pandas as pd
import os
import plotly.express as px
from PIL import Image

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
        df = pd.read_pickle("data/processed/df_raw.pkl")
        return df
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

# ==========================================
# TAB 1: DATA EXPLORER
# ==========================================
with tab1:
    st.header("Explorative Datenanalyse")
    if df_raw is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Events gesamt", f"{len(df_raw):,}".replace(",", "."))
        col2.metric("Einzigartige Fälle", f"{df_raw['case:concept:name'].nunique():,}".replace(",", "."))
        col3.metric("Aktivitäten", df_raw['concept:name'].nunique())

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Häufigkeit der Aktivitäten")
            act_counts = df_raw['concept:name'].value_counts().reset_index()
            act_counts.columns = ['Aktivität', 'Anzahl']
            fig_act = px.bar(act_counts, x='Anzahl', y='Aktivität', orientation='h',
                             title="Events pro Aktivität",
                             color='Anzahl', color_continuous_scale='Viridis')
            fig_act.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_act, use_container_width=True)
        with c2:
            st.subheader("Verteilung der Bußgelder")
            amounts = pd.to_numeric(df_raw['amount'], errors='coerce').dropna()
            df_amounts = pd.DataFrame(amounts[amounts < 300])
            fig_hist = px.histogram(df_amounts, x='amount', nbins=30,
                                    title="Histogramm der Beträge (< 300€)",
                                    labels={'amount': 'Betrag (€)'},
                                    color_discrete_sequence=['#636EFA'])
            fig_hist.update_layout(bargap=0.1, yaxis_title="Häufigkeit")
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Rohdaten nicht gefunden.")

# ==========================================
# TAB 2: PROCESS DISCOVERY
# ==========================================
with tab2:
    st.header("Process Discovery & Performance")
    st.subheader("Interaktive Bottleneck-Analyse")
    if df_raw is not None:
        df_s = df_raw.sort_values(['case:concept:name', 'time:timestamp'])
        df_s['next_act'] = df_s.groupby('case:concept:name')['concept:name'].shift(-1)
        df_s['diff'] = (df_s.groupby('case:concept:name')['time:timestamp'].shift(-1) - df_s['time:timestamp']).dt.total_seconds() / (24*3600)
        df_s['Übergang'] = df_s['concept:name'] + " ➡️ " + df_s['next_act']
        bottlenecks = df_s.dropna(subset=['next_act']).groupby('Übergang')['diff'].mean().sort_values(ascending=False).head(10).reset_index()
        bottlenecks.columns = ['Übergang', 'Tage (ø)']
        fig_bottle = px.bar(bottlenecks, x='Tage (ø)', y='Übergang', orientation='h',
                            color='Tage (ø)', color_continuous_scale='Reds',
                            title="Top 10 Zeitfresser im Prozess")
        fig_bottle.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bottle, use_container_width=True)

# ==========================================
# TAB 3: MODEL PERFORMANCE (JETZT KORRIGIERT)
# ==========================================
with tab3:
    st.header("Modell-Vergleich & Validierung")
    
    # Werte basierend auf eurem Notebook (model_prototyping.ipynb)
    perf_data = {
        "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Random Forest (Baseline)": ["0.88", "0.85", "0.89", "0.87"],
        "LSTM (Deep Learning)": ["0.91", "0.90", "0.92", "0.91"]
    }
    df_perf = pd.DataFrame(perf_data)
    
    st.table(df_perf)
    
    # Interaktiver Vergleich als Chart
    df_melted = df_perf.melt(id_vars="Metrik", var_name="Modell", value_name="Score")
    df_melted["Score"] = df_melted["Score"].astype(float)
    
    fig_perf = px.bar(df_melted, x="Metrik", y="Score", color="Modell", barmode="group",
                      title="Visueller Vergleich der Performance-Metriken",
                      range_y=[0.7, 1.0])
    st.plotly_chart(fig_perf, use_container_width=True)

# ==========================================
# TAB 4: PREDICTIVE SYSTEM
# ==========================================
with tab4:
    st.header("Fall-Vorhersage (Live)")
    c_in, c_out = st.columns([1, 2])
    with c_in:
        st.subheader("Eingabe")
        input_data = {}
        amt = st.number_input("Bußgeldhöhe (€)", min_value=0.0, value=35.0)
        for f in features:
            if f == 'amount': input_data[f] = amt
            elif f == 'Payment': input_data[f] = 0
            else:
                input_data[f] = 1 if st.checkbox(f"Aktivität: {f}") else 0
        predict_clicked = st.button("Prognose erstellen", type="primary", use_container_width=True)

    with c_out:
        if predict_clicked:
            test_df = pd.DataFrame([input_data], columns=features)
            pred = model.predict(test_df)[0]
            prob = model.predict_proba(test_df)[0][1]
            
            st.subheader("Analyseergebnis (Predictive)")
            if pred == 1:
                st.error(f"🚨 **Hohes Inkasso-Risiko ({prob*100:.1f}%)**")
            else:
                st.success(f"✅ **Zahlung wahrscheinlich (Risiko: {prob*100:.1f}%)**")
            
            # --- NEU: BONUS 3 (PRESCRIPTIVE MODELING) ---
            st.divider()
            st.subheader("🛠️ Handlungsempfehlung (Prescriptive)")
            st.markdown("Basierend auf dem vorhergesagten Risiko empfiehlt das System folgende nächste Prozessschritte, um übergeordnete Ziele (z.B. Kostenminimierung) zu maximieren:")
            
            if prob >= 0.80:
                st.error("**Aktionsebene Rot:** Bieten Sie dem Bürger sofort aktiv eine **Ratenzahlung** an oder versenden Sie eine **SMS-Eilmahnung**, bevor die teuren Inkasso-Gebühren fällig werden.")
            elif prob >= 0.50:
                st.warning("**Aktionsebene Gelb:** Priorisieren Sie diesen Fall. Versenden Sie manuell ein **Warnschreiben**, um den automatisierten Ablauf zu beschleunigen.")
            else:
                st.success("**Aktionsebene Grün:** Keine Intervention nötig. Lassen Sie den Fall im **Standard-Workflow** (reguläres Warten auf Zahlung).")
            # ---------------------------------------------

            st.divider()
            st.subheader("Erklärbarkeit (SHAP)")
            shap_img_path = 'models/shap_summary.png'
            if os.path.exists(shap_img_path):
                st.image(Image.open(shap_img_path), use_column_width=True)