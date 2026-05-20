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
        # --- SICHERHEITS-CHECK: Zeitstempel-Format erzwingen ---
        if not pd.api.types.is_datetime64_any_dtype(df_raw['time:timestamp']):
            df_raw['time:timestamp'] = pd.to_datetime(df_raw['time:timestamp'], errors='coerce')

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
            st.plotly_chart(fig_act, use_container_width=True, key="chart_activities")
            
        with c2:
            st.subheader("Verteilung der Bußgelder")
            amounts = pd.to_numeric(df_raw['amount'], errors='coerce').dropna()
            df_amounts = pd.DataFrame(amounts[amounts < 300])
            fig_hist = px.histogram(df_amounts, x='amount', nbins=30,
                                    title="Histogramm der Beträge (< 300€)",
                                    labels={'amount': 'Betrag (€)'},
                                    color_discrete_sequence=['#636EFA'])
            fig_hist.update_layout(bargap=0.1, yaxis_title="Häufigkeit")
            st.plotly_chart(fig_hist, use_container_width=True, key="chart_fines")

        st.divider()
        st.subheader("Prozess- und Zeit-Metriken")
        
        # Daten auf Fall-Ebene aggregieren
        case_stats = df_raw.groupby('case:concept:name').agg(
            start_time=('time:timestamp', 'min'),
            end_time=('time:timestamp', 'max'),
            event_count=('concept:name', 'count')
        ).reset_index()
        
        # Durchlaufzeit in Tagen berechnen
        case_stats['duration_days'] = (case_stats['end_time'] - case_stats['start_time']).dt.total_seconds() / (24*3600)

        c3, c4 = st.columns(2)
        with c3:
            # Plot 3: Events pro Fall
            fig_length = px.histogram(case_stats, x='event_count', nbins=15,
                                      title="Verteilung der Falllängen",
                                      labels={'event_count': 'Anzahl Events pro Fall'},
                                      color_discrete_sequence=['#00CC96'])
            fig_length.update_layout(bargap=0.1, yaxis_title="Anzahl Fälle")
            # HIER WURDE DER KEY HINZUGEFÜGT
            st.plotly_chart(fig_length, use_container_width=True, key="chart_case_length")
            
        with c4:
            # Plot 4: Durchlaufzeiten
            df_duration_filtered = case_stats[case_stats['duration_days'] < 1000]
            fig_duration = px.histogram(df_duration_filtered, x='duration_days', nbins=40,
                                        title="Durchlaufzeiten der Fälle (< 1000 Tage)",
                                        labels={'duration_days': 'Dauer (in Tagen)'},
                                        color_discrete_sequence=['#EF553B'])
            fig_duration.update_layout(bargap=0.1, yaxis_title="Anzahl Fälle")
            # HIER WURDE DER KEY HINZUGEFÜGT
            st.plotly_chart(fig_duration, use_container_width=True, key="chart_case_duration")

        st.divider()
        
        # Plot 5: Workload über die Zeit
        st.subheader("Zeitliche Verteilung des Fallaufkommens (Workload)")
        # Monat und Jahr extrahieren für eine saubere Zeitreihe
        df_raw['year_month'] = df_raw['time:timestamp'].dt.tz_localize(None).dt.to_period('M').dt.to_timestamp()
        workload = df_raw.groupby('year_month').size().reset_index(name='count')
        
        fig_workload = px.line(workload, x='year_month', y='count',
                               title="Anzahl der Events im Zeitverlauf (Monatsebene)",
                               labels={'year_month': 'Zeitpunkt', 'count': 'Anzahl Events'},
                               color_discrete_sequence=['#AB63FA'])
        fig_workload.update_traces(line=dict(width=3))
        # HIER WURDE DER KEY HINZUGEFÜGT
        st.plotly_chart(fig_workload, use_container_width=True, key="chart_workload")
        
    else:
        st.warning("Rohdaten nicht gefunden.")        

# ==========================================
# TAB 2: PROCESS DISCOVERY & EXPLORATION
# ==========================================
with tab2:
    st.header("Process Discovery & Performance")
    
    if df_raw is not None:
        # Daten einmalig für alle Graphen in diesem Tab sortieren
        df_s = df_raw.sort_values(['case:concept:name', 'time:timestamp'])

        # --- 1. Interaktive Bottleneck-Analyse ---
        st.subheader("⏳ Interaktive Bottleneck-Analyse")
        
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

        st.divider()

        # --- 2. INTERAKTIVES DOTTED CHART (PLOTLY) ---
        st.subheader("🔎 Interaktive Dotted Chart Analyse (Batching)")
        st.markdown("""
        Das **Dotted Chart** visualisiert Events über die Zeitachse. 
        * **Vertikale Muster:** Deuten auf **Batching** (Stapelverarbeitung) hin.
        * **Tipp:** Nutzt die Lupe oben rechts im Graph, um in bestimmte Monate reinzuzoomen!
        """)
        
        # WICHTIG: Sampling! Ein Browser stürzt ab, wenn wir 100.000 Punkte in Plotly laden.
        # Wir nehmen eine zufällige Stichprobe von 300 Fällen für eine saubere Visualisierung.
        sample_cases_dc = df_s['case:concept:name'].drop_duplicates().sample(300, random_state=42)
        df_dc = df_s[df_s['case:concept:name'].isin(sample_cases_dc)].copy()
        
        fig_dc = px.scatter(df_dc, 
                            x="time:timestamp", 
                            y="case:concept:name", 
                            color="concept:name",
                            hover_data=["amount"],
                            title="Dotted Chart (Stichprobe von 300 Fällen)",
                            labels={"time:timestamp": "Zeitpunkt", "case:concept:name": "Fall-ID", "concept:name": "Aktivität"})
        
        # Die Y-Achsen-Beschriftung (Fall-IDs) ausblenden, da es zu viele sind
        fig_dc.update_yaxes(showticklabels=False, title_text="Fälle (Cases)")
        fig_dc.update_traces(marker=dict(size=5, opacity=0.8))
        
        st.plotly_chart(fig_dc, use_container_width=True)

        st.divider()

        # --- 3. INTERAKTIVES PERFORMANCE SPECTRUM (PLOTLY) ---
        st.subheader("📊 Interaktives Performance Spectrum")
        st.markdown("""
        Das **Performance Spectrum** zeigt den Fluss einzelner Fälle über die wichtigsten Prozessschritte im Zeitverlauf.
        * **Senkrechte/Steile Linien:** Extrem schneller Übergang zwischen zwei Aktivitäten (oft am selben Tag).
        * **Langgezogene/Diagonale Linien:** Engpässe und hohe Wartezeiten (die Zeitachse wandert monate- oder jahrelang nach rechts).
        * **Lücken im Graph:** Deuten auf Phasen hin, in denen die Behörde bestimmte Schritte nicht ausgeführt hat.
        """)
        
        # Wir filtern auf die Top 5 Aktivitäten, damit der Graph lesbar bleibt
        top_acts = df_s['concept:name'].value_counts().head(5).index.tolist()
        sample_cases_ps = df_s['case:concept:name'].drop_duplicates().sample(80, random_state=42)
        
        df_ps = df_s[df_s['case:concept:name'].isin(sample_cases_ps) & df_s['concept:name'].isin(top_acts)].copy()
        df_ps = df_ps.sort_values(by=['case:concept:name', 'time:timestamp'])
        
        # Y-Achsen Sortierung erzwingen
        df_ps['Aktivität'] = pd.Categorical(df_ps['concept:name'], categories=top_acts, ordered=True)
        
        fig_ps = px.line(df_ps, 
                         x="time:timestamp", 
                         y="Aktivität", 
                         line_group="case:concept:name", 
                         color_discrete_sequence=['#a02c34'],
                         markers=True,
                         title="Performance Spectrum (Stichprobe von 80 Fällen)",
                         hover_data=["case:concept:name", "amount"])
        
        # Linien leicht transparent machen, damit man Überschneidungen sieht
        fig_ps.update_traces(line=dict(width=1, color='rgba(160, 44, 52, 0.4)'), 
                             marker=dict(size=6, opacity=0.8, color='#a02c34'))
        
        fig_ps.update_yaxes(categoryorder='array', categoryarray=top_acts[::-1])
        st.plotly_chart(fig_ps, use_container_width=True)
    else:
        st.warning("Rohdaten nicht gefunden.")

        st.divider()

# --- 4. INTERAKTIVES PETRI-NETZ (NATIVES RENDERING) ---
    st.subheader("🕸️ Formales Prozessmodell: Petri-Netz (Happy Path)")
    
    dot_path = 'models/discovery/petri_net.dot'
    if os.path.exists(dot_path):
        with open(dot_path, 'r', encoding='utf-8') as f:
            dot_code = f.read()
        
        # --- NEU: Standard-Farben (Corporate Red) erzwingen ---
        # Wir injizieren unser '#a02c34' direkt als Basis-Style in den Graphviz-Code
        style_injection = '\n  node [color="#a02c34", fontcolor="#a02c34", fontname="Arial"];\n  edge [color="#a02c34"];\n'
        
        if '{' in dot_code:
            # Finde die erste öffnende Klammer '{' des Graphen und füge die Farben direkt danach ein
            insert_index = dot_code.find('{') + 1
            dot_code = dot_code[:insert_index] + style_injection + dot_code[insert_index:]
            
        st.graphviz_chart(dot_code)
    else:
        st.info("💡 Der DOT-Quellcode wurde noch nicht gefunden. Bitte führt im Terminal einmalig `python src/evaluate.py` aus.")
# ==========================================
# TAB 3: MODEL PERFORMANCE
# ==========================================
with tab3:
    st.header("Modell-Vergleich & Validierung")
    
    # Werte basierend auf eurem Notebook
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
                st.image(Image.open(shap_img_path), use_container_width=True)