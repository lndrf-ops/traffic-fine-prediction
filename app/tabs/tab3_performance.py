import streamlit as st
import pandas as pd
import plotly.express as px

def render():
    st.header("Modell-Vergleich & Validierung")
    
    perf_data = {
        "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Random Forest (Baseline)": ["0.88", "0.85", "0.89", "0.87"],
        "LSTM (Deep Learning)": ["0.91", "0.90", "0.92", "0.91"]
    }
    df_perf = pd.DataFrame(perf_data)
    st.table(df_perf)
    
    df_melted = df_perf.melt(id_vars="Metrik", var_name="Modell", value_name="Score")
    df_melted["Score"] = df_melted["Score"].astype(float)
    
    fig_perf = px.bar(df_melted, x="Metrik", y="Score", color="Modell", barmode="group",
                      title="Visueller Vergleich der Performance-Metriken", range_y=[0.7, 1.0])
    st.plotly_chart(fig_perf, use_container_width=True, key="chart_model_performance")