import streamlit as st
import pandas as pd
import plotly.express as px

def render(df_raw):
    st.header("Explorative Datenanalyse")
    if df_raw is not None:
        if not pd.api.types.is_datetime64_any_dtype(df_raw['time:timestamp']):
            df_raw['time:timestamp'] = pd.to_datetime(df_raw['time:timestamp'], errors='coerce')

        col1, col2, col3 = st.columns(3)
        col1.metric("Events gesamt", f"{len(df_raw):,}".replace(",", "."))
        col2.metric("Einzigartige Fälle", f"{df_raw['case:concept:name'].nunique():,}".replace(",", "."))
        col3.metric("Aktivitäten", df_raw['concept:name'].nunique())

        st.divider()
        st.subheader("Top 5 Prozessvarianten")
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
        variant_counts.columns = ['Prozessvariante', 'Anzahl Fälle']
        fig_variants = px.bar(
            variant_counts,
            x='Anzahl Fälle',
            y='Prozessvariante',
            orientation='h',
            title='Top 5 häufigste Prozessvarianten',
            color='Anzahl Fälle',
            color_continuous_scale='Teal'
        )
        fig_variants.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_variants, use_container_width=True, key='chart_top_variants')

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Häufigkeit der Aktivitäten")
            act_counts = df_raw['concept:name'].value_counts().reset_index()
            act_counts.columns = ['Aktivität', 'Anzahl']
            fig_act = px.bar(act_counts, x='Anzahl', y='Aktivität', orientation='h',
                             title="Events pro Aktivität", color='Anzahl', color_continuous_scale='Viridis')
            fig_act.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_act, use_container_width=True, key="chart_activities")
            
        with c2:
            st.subheader("Verteilung der Bußgelder")
            amounts = pd.to_numeric(df_raw['amount'], errors='coerce').dropna()
            df_amounts = pd.DataFrame(amounts[amounts < 300])
            fig_hist = px.histogram(df_amounts, x='amount', nbins=30,
                                    title="Histogramm der Beträge (< 300€)", labels={'amount': 'Betrag (€)'},
                                    color_discrete_sequence=['#636EFA'])
            fig_hist.update_layout(bargap=0.1, yaxis_title="Häufigkeit")
            st.plotly_chart(fig_hist, use_container_width=True, key="chart_fines")

        st.divider()
        st.subheader("Prozess- und Zeit-Metriken")
        
        case_stats = df_raw.groupby('case:concept:name').agg(
            start_time=('time:timestamp', 'min'), end_time=('time:timestamp', 'max'), event_count=('concept:name', 'count')
        ).reset_index()
        case_stats['duration_days'] = (case_stats['end_time'] - case_stats['start_time']).dt.total_seconds() / (24*3600)

        c3, c4 = st.columns(2)
        with c3:
            fig_length = px.histogram(case_stats, x='event_count', nbins=15, title="Verteilung der Falllängen",
                                      labels={'event_count': 'Anzahl Events pro Fall'}, color_discrete_sequence=['#00CC96'])
            fig_length.update_layout(bargap=0.1, yaxis_title="Anzahl Fälle")
            st.plotly_chart(fig_length, use_container_width=True, key="chart_case_length")
            
        with c4:
            df_duration_filtered = case_stats[case_stats['duration_days'] < 1000]
            fig_duration = px.histogram(df_duration_filtered, x='duration_days', nbins=40, title="Durchlaufzeiten der Fälle (< 1000 Tage)",
                                        labels={'duration_days': 'Dauer (in Tagen)'}, color_discrete_sequence=['#EF553B'])
            fig_duration.update_layout(bargap=0.1, yaxis_title="Anzahl Fälle")
            st.plotly_chart(fig_duration, use_container_width=True, key="chart_case_duration")

        st.divider()
        st.subheader("Zeitliche Verteilung des Fallaufkommens (Workload)")
        df_raw['year_month'] = df_raw['time:timestamp'].dt.tz_localize(None).dt.to_period('M').dt.to_timestamp()
        workload = df_raw.groupby('year_month').size().reset_index(name='count')
        
        fig_workload = px.line(workload, x='year_month', y='count', title="Anzahl der Events im Zeitverlauf (Monatsebene)",
                               labels={'year_month': 'Zeitpunkt', 'count': 'Anzahl Events'}, color_discrete_sequence=['#AB63FA'])
        fig_workload.update_traces(line=dict(width=3))
        st.plotly_chart(fig_workload, use_container_width=True, key="chart_workload")
        
    else:
        st.warning("Rohdaten nicht gefunden.")