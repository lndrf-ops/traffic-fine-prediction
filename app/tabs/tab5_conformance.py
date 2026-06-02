import json
import os

import plotly.express as px
import pandas as pd
import streamlit as st


def render(conformance_results: dict | None):
    st.header("Conformance Checking")
    st.caption(
        "Domain-specific compliance rules derived from the Italian Codice della Strada (CdS). "
        "Token-Based Replay following van der Aalst (2016), Ch. 8."
    )

    if not conformance_results:
        st.warning("Conformance results not found. Run 'python run_pipeline.py' first.")
        return

    # --- Token-Based Replay ---
    st.subheader("Token-Based Replay Fitness")
    tbr = conformance_results.get("token_based_fitness", {})
    col1, col2 = st.columns(2)
    col1.metric(
        "% Fit Traces",
        f"{tbr.get('perc_fit_traces', 0):.1f}%",
        help="Fraction of traces that can be replayed without missing/remaining tokens.",
    )
    col2.metric(
        "Avg. Trace Fitness",
        f"{tbr.get('average_trace_fitness', 0):.4f}",
        help="Mean token-based fitness score across all replayed traces.",
    )
    st.caption(f"Method: {tbr.get('method', '')}  ·  Source: {tbr.get('source', '')}")

    st.divider()

    # --- Rule compliance bar chart ---
    st.subheader("Business Rule Compliance")

    rules_data = []
    for key in ["rule1", "rule2", "rule3", "rule4", "rule5"]:
        rule = conformance_results.get(key)
        if not rule:
            continue
        rules_data.append(
            {
                "Rule": rule["rule"],
                "Compliance Rate": rule["compliance_rate"],
                "Violations": rule.get("violations", 0),
                "Source": rule.get("source", ""),
            }
        )

    if rules_data:
        df_rules = pd.DataFrame(rules_data)
        fig = px.bar(
            df_rules,
            x="Compliance Rate",
            y="Rule",
            orientation="h",
            color="Compliance Rate",
            color_continuous_scale=["#EF553B", "#FFA15A", "#00CC96"],
            range_x=[0.85, 1.005],
            title="Compliance Rate per Rule (closer to 1.0 = fully compliant)",
            text=df_rules["Compliance Rate"].map(lambda v: f"{v:.2%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, width="stretch", key="chart_compliance")

    # --- Detail table ---
    st.subheader("Detailed Rule Results")

    for key in ["rule1", "rule2", "rule3", "rule4", "rule5"]:
        rule = conformance_results.get(key)
        if not rule:
            continue

        compliance = rule["compliance_rate"]
        violations = rule.get("violations", 0)
        icon = "✅" if compliance >= 1.0 else ("⚠️" if compliance >= 0.95 else "❌")

        with st.expander(f"{icon} {rule['rule']}  ({compliance:.2%} compliant)"):
            cols = st.columns(3)
            cols[0].metric("Compliance Rate", f"{compliance:.4f}")
            cols[1].metric("Violations", f"{violations:,}")

            # Show denominator depending on which key is available
            denom_key = next(
                (k for k in ["total_cases", "cases_with_appeal", "cases_with_credit_collection",
                              "applicable_cases"] if k in rule),
                None,
            )
            if denom_key:
                cols[2].metric(denom_key.replace("_", " ").title(), f"{rule[denom_key]:,}")

            st.caption(f"Source: {rule.get('source', 'N/A')}")
