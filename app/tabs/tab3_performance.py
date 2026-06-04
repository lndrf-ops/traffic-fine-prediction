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
    MODEL_ORDER = [
        "Majority Baseline (Control-Flow)", "Majority Baseline (Data-Aware)",
        "Logistic Regression (Control-Flow)", "Logistic Regression (Data-Aware)",
        "Linear Regression (Control-Flow)", "Linear Regression (Data-Aware)",
        "Random Forest (Control-Flow)", "Random Forest (Data-Aware)",
        "XGBoost (Control-Flow)", "XGBoost (Data-Aware)",
        "LSTM (Control-Flow)", "LSTM (Data-Aware)",
        "Mean Baseline (Control-Flow)", "Mean Baseline (Data-Aware)",
    ]

    def _sort_by_order(df):
        """Sort dataframe index by MODEL_ORDER."""
        order = {name: i for i, name in enumerate(MODEL_ORDER)}
        return df.iloc[sorted(range(len(df)), key=lambda i: order.get(df.index[i], 99))]

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
        "Task: Predict whether a case ends in **No Collection** (0) or "
        "**Send for Credit Collection** (1). Temporal split 64/16/20."
    )

    # Reference k selector
    ref_k = st.radio(
        "Prefix length (k)",
        options=sorted(outcome["k"].unique()),
        index=len(sorted(outcome["k"].unique())) - 1,
        horizontal=True,
        key="outcome_k",
    )

    outcome_k = outcome[outcome["k"] == ref_k].copy()
    outcome_k["label"] = outcome_k.apply(make_label, axis=1)

    # Sub-tabs for per-class comparison
    tab_overall, tab_collection, tab_payment = st.tabs(["Overall", "Collection (class 1)", "No Collection (class 0)"])

    with tab_overall:
        fig_f1 = px.bar(
            outcome_k.sort_values("f1_collection", ascending=True),
            x="f1_collection",
            y="label",
            orientation="h",
            color="variant",
            color_discrete_map={"cf": "#8AC2D1", "da": "#B02F2C"},
            title=f"F1 Score (Collection class) — k={ref_k}",
            labels={"f1_collection": "F1 Score", "label": ""},
            range_x=[0, 1],
        )
        fig_f1.update_layout(showlegend=False)
        st.plotly_chart(fig_f1, width="stretch", key="chart_f1_overall")

        # Overall metrics table
        overall_cols = ["label", "f1_collection", "precision_collection", "recall_collection", "auc_roc", "accuracy"]
        available = [c for c in overall_cols if c in outcome_k.columns]
        df_overall = outcome_k[available].copy().rename(columns={
            "label": "Model", "f1_collection": "F1 (Collection)",
            "precision_collection": "Precision", "recall_collection": "Recall",
            "auc_roc": "AUC-ROC", "accuracy": "Accuracy",
        })
        df_overall = _sort_by_order(df_overall.set_index("Model"))
        for col in df_overall.columns:
            df_overall[col] = df_overall[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
        st.dataframe(df_overall, width="stretch")

        # Confusion matrices (best model: XGBoost, k=5)
        with st.expander("Confusion Matrices (XGBoost, k=5)"):
            cm_col1, cm_col2 = st.columns(2)
            cm_cf = "outputs/plots/confusion_matrix_cf_k5.png"
            cm_da = "outputs/plots/confusion_matrix_da_k5.png"
            with cm_col1:
                if os.path.exists(cm_cf):
                    st.image(cm_cf, caption="Control-Flow", width=500)
            with cm_col2:
                if os.path.exists(cm_da):
                    st.image(cm_da, caption="Data-Aware", width=500)

    with tab_collection:
        fig_coll = px.bar(
            outcome_k.sort_values("f1_collection", ascending=True),
            x="f1_collection",
            y="label",
            orientation="h",
            color="variant",
            color_discrete_map={"cf": "#8AC2D1", "da": "#B02F2C"},
            title=f"F1 Score — Credit Collection — k={ref_k}",
            labels={"f1_collection": "F1", "label": ""},
            range_x=[0, 1],
        )
        fig_coll.update_layout(showlegend=False)
        st.plotly_chart(fig_coll, width="stretch", key="chart_f1_coll")

        coll_cols = ["label", "f1_collection", "precision_collection", "recall_collection"]
        available = [c for c in coll_cols if c in outcome_k.columns]
        df_coll = outcome_k[available].copy().rename(columns={"label": "Model", "f1_collection": "F1", "precision_collection": "Precision", "recall_collection": "Recall"})
        df_coll = _sort_by_order(df_coll.set_index("Model"))
        for col in df_coll.columns:
            df_coll[col] = df_coll[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
        st.dataframe(df_coll, width="stretch")
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
                color_discrete_map={"cf": "#8AC2D1", "da": "#B02F2C"},
                title=f"F1 Score — Payment — k={ref_k}",
                labels={"f1_payment": "F1", "label": ""},
                range_x=[0, 1],
            )
            fig_pay.update_layout(showlegend=False)
            st.plotly_chart(fig_pay, width="stretch", key="chart_f1_pay")

            df_pay = outcome_k[available].copy().rename(columns={"label": "Model", "f1_payment": "F1", "precision_payment": "Precision", "recall_payment": "Recall"})
            df_pay = _sort_by_order(df_pay.set_index("Model"))
            for col in df_pay.columns:
                df_pay[col] = df_pay[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
            st.dataframe(df_pay, width="stretch")
            st.caption("Precision = of cases predicted as payment, how many truly are. Recall = of actual payment cases, how many are correctly identified.")
        else:
            st.info("Payment class metrics not available. Re-run the evaluation: `python -m src.t6_evaluate`")

    st.caption(
        "**F1** = harmonic mean of Precision and Recall. "
        "**AUC-ROC** = ranking ability across all thresholds."
    )

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
        '</span>',
        unsafe_allow_html=True,
    )

    ref_k_r = st.radio(
        "Prefix length (k)",
        options=sorted(remaining["k"].unique()),
        index=len(sorted(remaining["k"].unique())) - 1,
        horizontal=True,
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
        color_discrete_map={"cf": "#8AC2D1", "da": "#B02F2C"},
        title=f"MAE (days) — k={ref_k_r}",
        labels={"mae_days": "MAE (days)", "label": ""},
    )
    fig_mae.update_layout(showlegend=False)
    st.plotly_chart(fig_mae, width="stretch", key="chart_mae")

    # MAE table
    mae_cols = ["label", "mae_days"]
    if "rmse_days" in remaining_k.columns:
        mae_cols.append("rmse_days")
    mae_df = remaining_k[mae_cols].copy()
    mae_df = mae_df.rename(columns={"label": "Model", "mae_days": "MAE (days)", "rmse_days": "RMSE (days)"})
    mae_df = _sort_by_order(mae_df.set_index("Model"))
    for col in mae_df.columns:
        mae_df[col] = mae_df[col].apply(lambda x: f"{x:.1f}" if isinstance(x, float) else x)
    st.dataframe(mae_df, width="stretch")

    # ─── Overfitting Assessment ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Overfitting Assessment")

    overfit_path = "outputs/reports/overfitting_assessment.json"
    if os.path.exists(overfit_path):
        with open(overfit_path) as f:
            overfit_data = json.load(f)

        # Filter out baselines (they can't overfit by definition)
        BASELINE_MODELS = {"majority", "mean"}
        trained_data = [r for r in overfit_data if r["model"] not in BASELINE_MODELS]

        # Summary metrics
        verdicts = [r["verdict"] for r in trained_data]
        n_healthy = sum(1 for v in verdicts if v == "healthy")
        n_shift = sum(1 for v in verdicts if v == "distribution_shift")
        n_mild = sum(1 for v in verdicts if v == "mild")
        n_overfit = sum(1 for v in verdicts if v == "overfit")

        # Summary section
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("✅ Healthy", n_healthy)
        col2.metric("🔄 Distribution Shift", n_shift)
        col3.metric("⚠️ Mild", n_mild)
        col4.metric("❌ Overfit", n_overfit)

        if n_overfit == 0 and n_mild == 0:
            st.success("All trained models generalize well — no overfitting detected.")
        elif n_overfit == 0:
            st.warning(f"{n_mild} model(s) show mild overfitting. No severe cases.")
        else:
            st.error(f"{n_overfit} model(s) show signs of severe overfitting.")

        if n_shift > 0:
            st.info(
                f"🔄 {n_shift} model(s) show **distribution shift** (test outperforms train). "
                "This is expected with temporal splits where the test period has different characteristics."
            )

        # Helper to format rows
        def _format_overfit_df(rows, task):
            df = pd.DataFrame(rows)
            df["Model"] = df.apply(
                lambda r: f"{model_names.get(r['model'], r['model'])} ({variant_names.get(r['variant'], r['variant'])})", axis=1
            )
            order = {name: i for i, name in enumerate(MODEL_ORDER)}
            df["_sort"] = df["Model"].map(lambda x: order.get(x, 99))
            df = df.sort_values(["_sort", "k"]).drop(columns=["_sort", "model", "variant", "task"], errors="ignore")
            if task == "outcome":
                df = df.rename(columns={
                    "k": "Prefix k", "train_auc_roc": "Train AUC-ROC",
                    "test_auc_roc": "Test AUC-ROC", "gap": "Gap",
                    "verdict": "Verdict",
                })
            else:
                df = df.rename(columns={
                    "k": "Prefix k", "train_mae_days": "Train MAE (days)",
                    "test_mae_days": "Test MAE (days)", "rel_gap": "Relative Gap",
                    "verdict": "Verdict",
                })
            cols = [c for c in df.columns if c == "Model"] + [c for c in df.columns if c != "Model"]
            return df[cols]

        # Tabs for outcome vs remaining
        tab_outcome, tab_remaining = st.tabs(["Outcome Prediction", "Remaining Time Prediction"])

        with tab_outcome:
            outcome_rows = [r for r in trained_data if r["task"] == "outcome"]
            if outcome_rows:
                st.dataframe(_format_overfit_df(outcome_rows, "outcome"), hide_index=True, use_container_width=True)

        with tab_remaining:
            remaining_rows = [r for r in trained_data if r["task"] == "remaining"]
            if remaining_rows:
                st.dataframe(_format_overfit_df(remaining_rows, "remaining"), hide_index=True, use_container_width=True)
    else:
        st.info("Overfitting assessment not found. Run the evaluation pipeline.")

    # ─── Global Feature Importance (SHAP) ────────────────────────────────────
    st.markdown("---")
    st.subheader("Global Feature Importance (SHAP)")
    st.caption("Which features matter most for predicting credit collection? XGBoost with TreeExplainer (exact Shapley values).")

    shap_variant = st.radio("Variant", ["DA (Data-Aware)", "CF (Control-Flow)"], horizontal=True, key="shap_variant")
    shap_var_key = "da" if shap_variant.startswith("DA") else "cf"

    shap_cols = st.columns(3)
    for col, k in zip(shap_cols, [2, 3, 5]):
        shap_path = f"outputs/plots/shap_{shap_var_key}_k{k}.png"
        with col:
            if os.path.exists(shap_path):
                st.image(shap_path, caption=f"k={k}")
            else:
                st.info(f"k={k} not found.")

    st.markdown("**Remaining Time Prediction**")
    shap_cols_rem = st.columns(3)
    for col, k in zip(shap_cols_rem, [2, 3, 5]):
        shap_path = f"outputs/plots/shap_remaining_{shap_var_key}_k{k}.png"
        with col:
            if os.path.exists(shap_path):
                st.image(shap_path, caption=f"k={k}")
            else:
                st.info(f"k={k} not found. Run pipeline step 6.4.")

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
