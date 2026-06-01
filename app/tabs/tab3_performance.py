"""Tab 3: Model Performance — clean overview of all models and results."""

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render(eval_results):
    st.header("Model Performance")

    if not eval_results:
        st.warning("Evaluation results not found. Run `python run_pipeline.py` first.")
        return

    df = pd.DataFrame(eval_results)
    outcome = df[df["task"] == "outcome"].copy()
    remaining = df[df["task"] == "remaining"].copy()

    # Human-readable names
    model_names = {
        "majority": "Majority Baseline", "logreg": "Logistic Regression",
        "rf": "Random Forest", "xgb": "XGBoost", "lstm": "LSTM",
        "mean": "Mean Baseline", "linreg": "Linear Regression",
        "rf_reg": "Random Forest", "xgb_reg": "XGBoost",
    }
    variant_names = {"cf": "Control-Flow", "da": "Data-Aware"}

    def make_label(row):
        return f"{model_names.get(row['model'], row['model'])} ({variant_names.get(row['variant'], row['variant'])})"

    # ─── Key Findings ───────────────────────────────────────────────────────
    st.markdown("---")
    best_outcome = outcome.loc[outcome["auc_roc"].idxmax()]
    best_remaining = remaining.loc[remaining["mae_days"].idxmin()]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Best Outcome Model",
            f"{model_names.get(best_outcome['model'], best_outcome['model'])} ({variant_names.get(best_outcome['variant'], best_outcome['variant'])}, k={best_outcome['k']})",
            f"AUC-ROC: {best_outcome['auc_roc']:.4f}",
        )
    with col2:
        st.metric(
            "Best Remaining Time Model",
            f"{model_names.get(best_remaining['model'], best_remaining['model'])} ({variant_names.get(best_remaining['variant'], best_remaining['variant'])}, k={best_remaining['k']})",
            f"MAE: {best_remaining['mae_days']:.1f} days",
        )

    # ─── Outcome Prediction ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Outcome Prediction (Classification)")
    st.caption(
        "Task: Predict whether a case ends in **Payment** (0) or "
        "**Send for Credit Collection** (1). Temporal split 64/16/20."
    )

    # Reference k selector
    ref_k = st.select_slider(
        "Reference prefix length",
        options=sorted(outcome["k"].unique()),
        value=5,
        key="outcome_k",
    )

    outcome_k = outcome[outcome["k"] == ref_k].copy()
    outcome_k["label"] = outcome_k.apply(make_label, axis=1)

    # Sub-tabs for per-class comparison
    tab_overall, tab_collection, tab_payment = st.tabs(["Overall", "Collection (class 1)", "Payment (class 0)"])

    with tab_overall:
        fig_f1 = px.bar(
            outcome_k.sort_values("f1_collection", ascending=True),
            x="f1_collection",
            y="label",
            orientation="h",
            color="variant",
            color_discrete_map={"cf": "#636EFA", "da": "#EF553B"},
            title=f"F1 Score (Collection class) — k={ref_k}",
            labels={"f1_collection": "F1 Score", "label": ""},
            range_x=[0, 1],
        )
        fig_f1.update_layout(showlegend=False)
        st.plotly_chart(fig_f1, use_container_width=True, key="chart_f1_overall")

        # Overall metrics table
        overall_cols = ["label", "accuracy", "auc_roc"]
        available = [c for c in overall_cols if c in outcome_k.columns]
        df_overall = outcome_k[available].copy().rename(columns={"label": "Model", "accuracy": "Accuracy", "auc_roc": "AUC-ROC"})
        df_overall = df_overall.set_index("Model").sort_values("AUC-ROC", ascending=False)
        for col in df_overall.columns:
            df_overall[col] = df_overall[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
        st.dataframe(df_overall, use_container_width=True)

    with tab_collection:
        fig_coll = px.bar(
            outcome_k.sort_values("f1_collection", ascending=True),
            x="f1_collection",
            y="label",
            orientation="h",
            color="variant",
            color_discrete_map={"cf": "#636EFA", "da": "#EF553B"},
            title=f"F1 Score — Credit Collection — k={ref_k}",
            labels={"f1_collection": "F1", "label": ""},
            range_x=[0, 1],
        )
        fig_coll.update_layout(showlegend=False)
        st.plotly_chart(fig_coll, use_container_width=True, key="chart_f1_coll")

        coll_cols = ["label", "f1_collection", "precision_collection", "recall_collection"]
        available = [c for c in coll_cols if c in outcome_k.columns]
        df_coll = outcome_k[available].copy().rename(columns={"label": "Model", "f1_collection": "F1", "precision_collection": "Precision", "recall_collection": "Recall"})
        df_coll = df_coll.set_index("Model").sort_values("F1", ascending=False)
        for col in df_coll.columns:
            df_coll[col] = df_coll[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
        st.dataframe(df_coll, use_container_width=True)
        st.caption("Precision = of flagged cases, how many are truly collection. Recall = of actual collection cases, how many are caught.")

    with tab_payment:
        pay_cols = ["label", "f1_payment", "precision_payment", "recall_payment"]
        available = [c for c in pay_cols if c in outcome_k.columns]
        if len(available) > 1:
            fig_pay = px.bar(
                outcome_k.sort_values("f1_payment", ascending=True),
                x="f1_payment",
                y="label",
                orientation="h",
                color="variant",
                color_discrete_map={"cf": "#636EFA", "da": "#EF553B"},
                title=f"F1 Score — Payment — k={ref_k}",
                labels={"f1_payment": "F1", "label": ""},
                range_x=[0, 1],
            )
            fig_pay.update_layout(showlegend=False)
            st.plotly_chart(fig_pay, use_container_width=True, key="chart_f1_pay")

            df_pay = outcome_k[available].copy().rename(columns={"label": "Model", "f1_payment": "F1", "precision_payment": "Precision", "recall_payment": "Recall"})
            df_pay = df_pay.set_index("Model").sort_values("F1", ascending=False)
            for col in df_pay.columns:
                df_pay[col] = df_pay[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
            st.dataframe(df_pay, use_container_width=True)
            st.caption("Precision = of cases predicted as payment, how many truly are. Recall = of actual payment cases, how many are correctly identified.")
        else:
            st.info("Payment class metrics not available. Re-run the evaluation: `python -m src.t6_evaluate`")

    st.caption(
        "**F1** = harmonic mean of Precision and Recall. "
        "**AUC-ROC** = ranking ability across all thresholds."
    )

    # Detailed: all k values
    with st.expander("📋 Detailed: All prefix lengths"):
        pivot = outcome.pivot_table(
            index=["model", "variant"], columns="k", values="f1_collection"
        ).round(4)
        pivot.columns = [f"k={c}" for c in pivot.columns]
        pivot.index = [f"{m.upper()} ({v.upper()})" for m, v in pivot.index]
        st.dataframe(pivot, use_container_width=True)

    # ─── Remaining Time Prediction ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("⏱️ Remaining Time Prediction (Regression)")
    st.caption(
        "Task: Predict how many **days** remain until the case is closed. "
        "Lower MAE = better."
    )
    st.markdown(
        '<span title="MAE = Mean Absolute Error: average number of days the prediction is off. '
        'RMSE = Root Mean Squared Error: penalizes large errors more heavily than MAE.">'
        '💡 Hover for metric explanation</span>',
        unsafe_allow_html=True,
    )

    ref_k_r = st.select_slider(
        "Reference prefix length",
        options=sorted(remaining["k"].unique()),
        value=5,
        key="remaining_k",
    )

    remaining_k = remaining[remaining["k"] == ref_k_r].copy()
    remaining_k["label"] = remaining_k.apply(make_label, axis=1)

    fig_mae = px.bar(
        remaining_k.sort_values("mae_days", ascending=False),
        x="mae_days",
        y="label",
        orientation="h",
        color="variant",
        color_discrete_map={"cf": "#636EFA", "da": "#EF553B"},
        title=f"MAE (days) — k={ref_k_r}",
        labels={"mae_days": "MAE (days)", "label": ""},
    )
    fig_mae.update_layout(showlegend=False)
    st.plotly_chart(fig_mae, use_container_width=True, key="chart_mae")

    # MAE table
    mae_cols = ["label", "mae_days"]
    if "rmse_days" in remaining_k.columns:
        mae_cols.append("rmse_days")
    mae_df = remaining_k[mae_cols].copy()
    mae_df = mae_df.rename(columns={"label": "Model", "mae_days": "MAE (days)", "rmse_days": "RMSE (days)"})
    mae_df = mae_df.set_index("Model").sort_values("MAE (days)")
    for col in mae_df.columns:
        mae_df[col] = mae_df[col].apply(lambda x: f"{x:.1f}" if isinstance(x, float) else x)
    st.dataframe(mae_df, use_container_width=True)

    with st.expander("📋 Detailed: All prefix lengths"):
        pivot_r = remaining.pivot_table(
            index=["model", "variant"], columns="k", values="mae_days"
        ).round(1)
        pivot_r.columns = [f"k={c}" for c in pivot_r.columns]
        pivot_r.index = [f"{m.upper()} ({v.upper()})" for m, v in pivot_r.index]
        st.dataframe(pivot_r, use_container_width=True)

    # ─── Overfitting Assessment ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Overfitting Assessment")

    overfit_path = "outputs/reports/overfitting_assessment.json"
    if os.path.exists(overfit_path):
        with open(overfit_path) as f:
            overfit_data = json.load(f)

        verdicts = [r["verdict"] for r in overfit_data]
        n_ok = verdicts.count("ok")
        n_mild = verdicts.count("mild")
        n_overfit = verdicts.count("overfit")

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ OK", n_ok)
        col2.metric("⚠️ Mild", n_mild)
        col3.metric("❌ Overfit", n_overfit)

        if n_overfit == 0:
            st.success("No severe overfitting detected across any model configuration.")
        else:
            st.error(f"{n_overfit} model(s) show signs of overfitting.")

        with st.expander("Details: Mild overfitting cases"):
            mild = [r for r in overfit_data if r["verdict"] == "mild"]
            if mild:
                st.dataframe(pd.DataFrame(mild), use_container_width=True)
            else:
                st.info("No mild cases.")
    else:
        st.info("Overfitting assessment not found. Run the evaluation pipeline.")

    # ─── Metric Explanation ─────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("ℹ️ Metric Explanation"):
        st.markdown("""
| Metric | What it measures | Note |
|--------|-----------------|------|
| **F1 (Collection)** | Harmonic mean of Precision & Recall for the deviant class | Primary metric — balances false alarms vs. missed cases |
| **Precision** | Of cases flagged as collection, how many actually are | High = few false alarms |
| **Recall** | Of actual collection cases, how many are caught | High = few missed cases |
| **AUC-ROC** | Ranking ability across all decision thresholds | Threshold-independent, good for model comparison |
| **Accuracy** | Overall correct predictions | Misleading with class imbalance (majority gets ~60%) |
| **MAE** | Average absolute error in predicted remaining days | Lower = better |
        """)
