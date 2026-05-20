import streamlit as st
import pandas as pd
import plotly.express as px
import os

def render(df_raw):
    st.header("Process Discovery & Performance")
    
    if df_raw is not None:
        df_s = df_raw.sort_values(['case:concept:name', 'time:timestamp'])

        st.subheader("⏳ Interaktive Bottleneck-Analyse")
        df_s['next_act'] = df_s.groupby('case:concept:name')['concept:name'].shift(-1)
        df_s['diff'] = (df_s.groupby('case:concept:name')['time:timestamp'].shift(-1) - df_s['time:timestamp']).dt.total_seconds() / (24*3600)
        df_s['Übergang'] = df_s['concept:name'] + " ➡️ " + df_s['next_act']
        bottlenecks = df_s.dropna(subset=['next_act']).groupby('Übergang')['diff'].mean().sort_values(ascending=False).head(10).reset_index()
        bottlenecks.columns = ['Übergang', 'Tage (ø)']
        
        fig_bottle = px.bar(bottlenecks, x='Tage (ø)', y='Übergang', orientation='h', color='Tage (ø)', color_continuous_scale='Reds', title="Top 10 Zeitfresser im Prozess")
        fig_bottle.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bottle, use_container_width=True, key="chart_bottlenecks")

        st.divider()

        st.subheader("🔎 Interaktive Dotted Chart Analyse (Batching)")
        st.markdown("Das **Dotted Chart** visualisiert Events über die Zeitachse.\n* **Vertikale Muster:** Deuten auf **Batching** hin.")
        
        sample_cases_dc = df_s['case:concept:name'].drop_duplicates().sample(300, random_state=42)
        df_dc = df_s[df_s['case:concept:name'].isin(sample_cases_dc)].copy()
        fig_dc = px.scatter(df_dc, x="time:timestamp", y="case:concept:name", color="concept:name", hover_data=["amount"], title="Dotted Chart (Stichprobe von 300 Fällen)")
        fig_dc.update_yaxes(showticklabels=False, title_text="Fälle (Cases)")
        fig_dc.update_traces(marker=dict(size=5, opacity=0.8))
        st.plotly_chart(fig_dc, use_container_width=True, key="chart_dotted_chart")

        st.divider()

        st.subheader("📊 Interaktives Performance Spectrum")
        st.markdown("Das **Performance Spectrum** zeigt den Fluss einzelner Fälle.\n* **Senkrechte Linien:** Schnell.\n* **Diagonale Linien:** Engpässe.")
        
        top_acts = df_s['concept:name'].value_counts().head(5).index.tolist()
        sample_cases_ps = df_s['case:concept:name'].drop_duplicates().sample(80, random_state=42)
        df_ps = df_s[df_s['case:concept:name'].isin(sample_cases_ps) & df_s['concept:name'].isin(top_acts)].copy()
        df_ps = df_ps.sort_values(by=['case:concept:name', 'time:timestamp'])
        df_ps['Aktivität'] = pd.Categorical(df_ps['concept:name'], categories=top_acts, ordered=True)
        
        fig_ps = px.line(df_ps, x="time:timestamp", y="Aktivität", line_group="case:concept:name", color_discrete_sequence=['#a02c34'], markers=True, title="Performance Spectrum")
        fig_ps.update_traces(line=dict(width=1, color='rgba(160, 44, 52, 0.4)'), marker=dict(size=6, opacity=0.8, color='#a02c34'))
        fig_ps.update_yaxes(categoryorder='array', categoryarray=top_acts[::-1])
        st.plotly_chart(fig_ps, use_container_width=True, key="chart_performance_spectrum")
    else:
        st.warning("Rohdaten nicht gefunden.")

    st.divider()
    st.subheader("🕸️ Formales Prozessmodell: Petri-Netz (Happy Path)")
    
    dot_path = 'models/discovery/petri_net.dot'
    if os.path.exists(dot_path):
        with open(dot_path, 'r', encoding='utf-8') as f:
            dot_code = f.read()
        st.graphviz_chart(dot_code)
    else:
        st.info("💡 Der DOT-Quellcode wurde noch nicht gefunden. Bitte führt im Terminal einmalig `python src/evaluate.py` aus.")