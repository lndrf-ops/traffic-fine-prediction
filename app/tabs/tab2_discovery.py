import streamlit as st
import pandas as pd
import plotly.express as px
from theme import COLOR_PALETTE
import os

def render(df_raw):
    st.header("Process Discovery & Performance")
    
    if df_raw is not None:
        df_s = df_raw.sort_values(['case:concept:name', 'time:timestamp'])

        st.subheader("⏳ Interactive Bottleneck Analysis")
        df_s['next_act'] = df_s.groupby('case:concept:name')['concept:name'].shift(-1)
        df_s['diff'] = (df_s.groupby('case:concept:name')['time:timestamp'].shift(-1) - df_s['time:timestamp']).dt.total_seconds() / (24*3600)
        df_s['Transition'] = df_s['concept:name'] + " → " + df_s['next_act']
        bottleneck_stats = df_s.dropna(subset=['next_act']).groupby('Transition').agg(
            **{
                'Days_mean': ('diff', 'mean'),
                'Days_median': ('diff', 'median'),
                'Cases': ('case:concept:name', 'nunique')
            }
        ).reset_index()

        bottlenecks_mean = bottleneck_stats.sort_values('Days_mean', ascending=False).head(10).reset_index(drop=True)
        bottlenecks_median = bottleneck_stats.sort_values('Days_median', ascending=False).head(10).reset_index(drop=True)

        fig_bottle_mean = px.bar(
            bottlenecks_mean,
            x='Days_mean',
            y='Transition',
            orientation='h',
            color='Days_mean',
            hover_data=['Cases'],
            color_continuous_scale=[[0, COLOR_PALETTE[4]], [1, COLOR_PALETTE[0]]],
            title="Top 10 Bottlenecks (Mean)"
        )
        fig_bottle_mean.update_layout(yaxis={'categoryorder':'total ascending'})
        fig_bottle_mean.update_traces(hovertemplate='%{y}<br>Days (avg): %{x:.2f}<br>Cases: %{customdata[0]}<extra></extra>')
        st.plotly_chart(fig_bottle_mean, width="stretch", key="chart_bottlenecks_mean")

        fig_bottle_median = px.bar(
            bottlenecks_median,
            x='Days_median',
            y='Transition',
            orientation='h',
            color='Days_median',
            hover_data=['Cases'],
            color_continuous_scale=[[0, COLOR_PALETTE[5]], [1, COLOR_PALETTE[1]]],
            title="Top 10 Bottlenecks (Median)"
        )
        fig_bottle_median.update_layout(yaxis={'categoryorder':'total ascending'})
        fig_bottle_median.update_traces(hovertemplate='%{y}<br>Median days: %{x:.2f}<br>Cases: %{customdata[0]}<extra></extra>')
        st.plotly_chart(fig_bottle_median, width="stretch", key="chart_bottlenecks_median")

        st.divider()

        st.subheader("🔎 Interactive Dotted Chart (Batching Analysis)")
        st.markdown("The **Dotted Chart** visualizes events along the time axis.\n* **Vertical patterns:** Indicate **batching** behavior.")
        
        sample_cases_dc = df_s['case:concept:name'].drop_duplicates().sample(1000, random_state=42)
        df_dc = df_s[df_s['case:concept:name'].isin(sample_cases_dc)].copy()
        fig_dc = px.scatter(df_dc, x="time:timestamp", y="case:concept:name", color="concept:name", hover_data=["amount"], title="Dotted Chart (sample of 1000 cases)")
        fig_dc.update_yaxes(showticklabels=False, title_text="Cases")
        fig_dc.update_traces(marker=dict(size=5, opacity=0.8))
        st.plotly_chart(fig_dc, width="stretch", key="chart_dotted_chart")

        st.divider()

        st.subheader("📊 Interactive Performance Spectrum")
        st.markdown("The **Performance Spectrum** shows the flow of individual cases.\n* **Vertical lines:** Fast transitions.\n* **Diagonal lines:** Bottlenecks.")
        
        top_acts = df_s['concept:name'].value_counts().head(5).index.tolist()
        sample_cases_ps = df_s['case:concept:name'].drop_duplicates().sample(80, random_state=42)
        df_ps = df_s[df_s['case:concept:name'].isin(sample_cases_ps) & df_s['concept:name'].isin(top_acts)].copy()
        df_ps = df_ps.sort_values(by=['case:concept:name', 'time:timestamp'])
        df_ps['Activity'] = pd.Categorical(df_ps['concept:name'], categories=top_acts, ordered=True)
        
        fig_ps = px.line(df_ps, x="time:timestamp", y="Activity", line_group="case:concept:name", color_discrete_sequence=[COLOR_PALETTE[0]], markers=True, title="Performance Spectrum")
        fig_ps.update_traces(line=dict(width=1, color='rgba(176, 47, 44, 0.4)'), marker=dict(size=6, opacity=0.8, color=COLOR_PALETTE[0]))
        fig_ps.update_yaxes(categoryorder='array', categoryarray=top_acts[::-1])
        st.plotly_chart(fig_ps, width="stretch", key="chart_performance_spectrum")
    else:
        st.warning("Raw data not found.")

    st.divider()
    st.subheader("🕸️ Formal Process Model: Petri Net (Happy Path)")
    
    dot_path = 'outputs/plots/petri_net.dot'
    if os.path.exists(dot_path):
        with open(dot_path, 'r', encoding='utf-8') as f:
            dot_code = f.read()
        st.graphviz_chart(dot_code)
    else:
        st.info("💡 Petri net not found. Please run `python run_pipeline.py` first.")