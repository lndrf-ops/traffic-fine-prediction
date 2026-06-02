"""Tab 6: Generative AI Evaluation — inspect synthetic trace quality and downloads."""

import json
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st


def _trace_lengths_from_completed(completed_cases: pd.DataFrame):
    if completed_cases is None:
        return None
    if 'trace' in completed_cases.columns:
        return completed_cases['trace'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    if 'concept:name' in completed_cases.columns:
        return completed_cases.groupby('case:concept:name')['concept:name'].size()
    return None


def _trace_lengths_from_synthetic(synthetic_log: pd.DataFrame):
    if synthetic_log is None:
        return None
    return synthetic_log.groupby('case:concept:name')['concept:name'].size()


def _top_df_pairs_from_trace_list(traces):
    pairs = Counter()
    for trace in traces:
        for i in range(len(trace) - 1):
            pairs[(trace[i], trace[i + 1])] += 1
    return pairs.most_common(10)


def _top_df_pairs_from_log(df):
    if df is None or df.empty:
        return []
    pairs = Counter()
    for case_id, group in df.groupby('case:concept:name'):
        events = group['concept:name'].tolist()
        for i in range(len(events) - 1):
            pairs[(events[i], events[i + 1])] += 1
    return pairs.most_common(10)


def render(generative_results, synthetic_log, completed_cases):
    st.header("Generative AI Evaluation")

    if not generative_results:
        st.warning("Generative quality report not found. Run `python src/t8_generative_ai.py` first.")
        return

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Activity JSD", f"{generative_results['activity_jsd']:.4f}")
    col2.metric("DF Coverage", f"{generative_results['directly_follows_coverage']:.1%}")
    col3.metric(
        "Real trace length",
        f"{generative_results['real_trace_length_mean']:.1f} ± {generative_results['real_trace_length_std']:.1f}",
    )
    col4.metric(
        "Synthetic trace length",
        f"{generative_results['synth_trace_length_mean']:.1f} ± {generative_results['synth_trace_length_std']:.1f}",
    )

    st.markdown("---")

    # Activity distribution
    activity_df = pd.DataFrame.from_records(
        [
            {
                "activity": activity,
                "real_pct": values["real_pct"],
                "synthetic_pct": values["synth_pct"],
            }
            for activity, values in generative_results.get("activity_distribution", {}).items()
        ]
    )
    if not activity_df.empty:
        activity_df = activity_df.sort_values("real_pct", ascending=False)
        fig_activity = px.bar(
            activity_df,
            x="activity",
            y=["real_pct", "synthetic_pct"],
            barmode="group",
            title="Activity Distribution: Real vs Synthetic",
            labels={"value": "Share (%)", "activity": "Activity"},
            height=450,
        )
        st.plotly_chart(fig_activity, use_container_width=True)

    # Trace length comparison
    real_lengths = _trace_lengths_from_completed(completed_cases)
    synth_lengths = _trace_lengths_from_synthetic(synthetic_log)
    if real_lengths is not None and synth_lengths is not None and not synth_lengths.empty:
        len_df = pd.DataFrame(
            {
                "real": real_lengths,
                "synthetic": synth_lengths,
            }
        ).melt(var_name="source", value_name="trace_length")

        fig_len = px.histogram(
            len_df,
            x="trace_length",
            color="source",
            barmode="group",
            nbins=20,
            title="Trace Length Distribution: Real vs Synthetic",
            labels={"trace_length": "Trace Length", "source": "Source"},
            histnorm="probability",
        )
        st.plotly_chart(fig_len, use_container_width=True)
    else:
        st.info("Trace length comparison is not available because real or synthetic trace data is missing.")

    # Top directly-follows pairs
    st.markdown("---")
    st.subheader("Top directly-follows pairs")

    real_pairs = []
    if completed_cases is not None and 'trace' in completed_cases.columns:
        real_pairs = _top_df_pairs_from_trace_list(completed_cases['trace'].tolist())

    synth_pairs = _top_df_pairs_from_log(synthetic_log)

    pair_col1, pair_col2 = st.columns(2)
    with pair_col1:
        st.markdown("**Real log**")
        if real_pairs:
            st.write(pd.DataFrame(real_pairs, columns=["pair", "count"]))
        else:
            st.info("Real directly-follows pairs not available.")

    with pair_col2:
        st.markdown("**Synthetic log**")
        if synth_pairs:
            st.write(pd.DataFrame(synth_pairs, columns=["pair", "count"]))
        else:
            st.info("Synthetic directly-follows pairs not available.")

    # Sample traces
    st.markdown("---")
    st.subheader("Sample traces")
    sample_real, sample_synth = st.columns(2)
    with sample_real:
        st.markdown("**Real traces**")
        if completed_cases is not None and 'trace' in completed_cases.columns:
            for i, trace in enumerate(completed_cases['trace'].head(5).tolist(), 1):
                st.write(f"{i}. {trace}")
        else:
            st.info("Real traces not available.")

    with sample_synth:
        st.markdown("**Synthetic traces**")
        if synthetic_log is not None and not synthetic_log.empty:
            syn_groups = synthetic_log.groupby('case:concept:name')['concept:name'].apply(list)
            for i, trace in enumerate(syn_groups.head(5).tolist(), 1):
                st.write(f"{i}. {trace}")
        else:
            st.info("Synthetic traces not available.")
