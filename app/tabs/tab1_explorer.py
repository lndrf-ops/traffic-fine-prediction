import streamlit as st
import pandas as pd
import plotly.express as px

def render(df_raw):
    st.header("Exploratory Data Analysis")
    if df_raw is not None:
        if not pd.api.types.is_datetime64_any_dtype(df_raw['time:timestamp']):
            df_raw['time:timestamp'] = pd.to_datetime(df_raw['time:timestamp'], errors='coerce')

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", f"{len(df_raw):,}")
        col2.metric("Unique Cases", f"{df_raw['case:concept:name'].nunique():,}")
        col3.metric("Activities", df_raw['concept:name'].nunique())

        st.divider()
        st.subheader("Top 5 Process Variants")
        df_variants = (
            df_raw.sort_values(['case:concept:name', 'time:timestamp'])
                  .groupby('case:concept:name')['concept:name']
                  .agg(lambda x: ' ➜ '.join(x))
                  .reset_index()
        )
        variant_counts = (
            df_variants['concept:name']
                       .value_counts()
                       .head(5)
                       .reset_index()
        )
        variant_counts.columns = ['Process Variant', 'Case Count']
        fig_variants = px.bar(
            variant_counts,
            x='Case Count',
            y='Process Variant',
            orientation='h',
            title='Top 5 Most Frequent Process Variants',
            color='Case Count',
            color_continuous_scale='Teal'
        )
        fig_variants.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_variants, use_container_width=True, key='chart_top_variants')

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Activity Frequency")
            act_counts = df_raw['concept:name'].value_counts().reset_index()
            act_counts.columns = ['Activity', 'Count']
            fig_act = px.bar(act_counts, x='Count', y='Activity', orientation='h',
                             title="Events per Activity", color='Count', color_continuous_scale='Viridis')
            fig_act.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_act, use_container_width=True, key="chart_activities")
            
        with c2:
            st.subheader("Fine Amount Distribution")
            amounts = pd.to_numeric(df_raw['amount'], errors='coerce').dropna()
            df_amounts = pd.DataFrame(amounts[amounts < 300])
            fig_hist = px.histogram(df_amounts, x='amount', nbins=30,
                                    title="Fine Amounts (< 300€)", labels={'amount': 'Amount (€)'},
                                    color_discrete_sequence=['#636EFA'])
            fig_hist.update_layout(bargap=0.1, yaxis_title="Frequency")
            st.plotly_chart(fig_hist, use_container_width=True, key="chart_fines")

        st.divider()
        st.subheader("Process & Time Metrics")
        
        case_stats = df_raw.groupby('case:concept:name').agg(
            start_time=('time:timestamp', 'min'), end_time=('time:timestamp', 'max'), event_count=('concept:name', 'count')
        ).reset_index()
        case_stats['duration_days'] = (case_stats['end_time'] - case_stats['start_time']).dt.total_seconds() / (24*3600)

        c3, c4 = st.columns(2)
        with c3:
            fig_length = px.histogram(case_stats, x='event_count', nbins=15, title="Case Length Distribution",
                                      labels={'event_count': 'Events per Case'}, color_discrete_sequence=['#00CC96'])
            fig_length.update_layout(bargap=0.1, yaxis_title="Number of Cases")
            st.plotly_chart(fig_length, use_container_width=True, key="chart_case_length")
            
        with c4:
            df_duration_filtered = case_stats[case_stats['duration_days'] < 1000]
            fig_duration = px.histogram(df_duration_filtered, x='duration_days', nbins=40, title="Case Duration (< 1000 days)",
                                        labels={'duration_days': 'Duration (days)'}, color_discrete_sequence=['#EF553B'])
            fig_duration.update_layout(bargap=0.1, yaxis_title="Number of Cases")
            st.plotly_chart(fig_duration, use_container_width=True, key="chart_case_duration")

        st.divider()
        st.subheader("Event Workload Over Time")
        df_raw['year_month'] = df_raw['time:timestamp'].dt.tz_localize(None).dt.to_period('M').dt.to_timestamp()
        workload = df_raw.groupby('year_month').size().reset_index(name='count')
        
        fig_workload = px.line(workload, x='year_month', y='count', title="Monthly Event Count",
                               labels={'year_month': 'Time', 'count': 'Event Count'}, color_discrete_sequence=['#AB63FA'])
        fig_workload.update_traces(line=dict(width=3))
        st.plotly_chart(fig_workload, use_container_width=True, key="chart_workload")
        
    else:
        st.warning("Raw data not found.")