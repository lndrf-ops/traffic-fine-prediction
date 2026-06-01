"""Tab 4: Live Case Prediction."""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ACTIVITIES = [
    "Create Fine",
    "Send Fine",
    "Insert Fine Notification",
    "Add penalty",
    "Insert Date Appeal to Prefecture",
    "Send Appeal to Prefecture",
    "Receive Result Appeal from Prefecture",
    "Notify Result Appeal to Offender",
    "Appeal to Judge",
]

# Available prefix lengths (must match trained models)
AVAILABLE_K = [2, 3, 5, 8]


def _best_k(n_events: int) -> int | None:
    """Select the largest trained k that is <= n_events."""
    valid = [k for k in AVAILABLE_K if k <= n_events]
    return max(valid) if valid else None


def render(models: dict):
    st.header("Live Case Prediction")

    st.markdown(
        "Simulate a running case by selecting which activities have occurred so far. "
        "The model predicts the probability that this case will end in **credit collection** "
        "based on the activity pattern observed up to this point."
    )
    st.caption(
        "The model uses binary features (activity observed: yes/no) from cases with similar "
        "prefix lengths — it does NOT filter to cases with exactly these activities."
    )

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.subheader("Case Configuration")

        st.markdown("**Activities observed so far:**")
        prefix_events = st.multiselect(
            "Select activities that have occurred in this case:",
            options=ACTIVITIES,
            default=["Create Fine", "Send Fine"],
        )

        n_events = len(prefix_events)
        selected_k = _best_k(n_events)

        if selected_k is None:
            st.warning(
                f"Select at least **{min(AVAILABLE_K)}** activities to enable prediction."
            )
        else:
            st.info(f"Using model trained on prefixes of length **k={selected_k}**")

        st.divider()

        variant = st.radio(
            "Feature variant",
            ["Control-Flow only", "Data-Aware (+ temporal, amount, …)"],
            help="CF uses only binary activity indicators. DA adds prefix length, duration, amount, points, and other case attributes.",
        )
        variant_key = "cf" if variant.startswith("Control") else "da"

        duration_days = 0
        amount = 35.0
        points = 0
        if variant_key == "da":
            st.markdown("**Additional attributes:**")
            duration_days = st.number_input(
                "Days since case start", min_value=0, max_value=2000, value=0, step=7,
                help="How many days have elapsed since the first event?",
            )
            amount = st.number_input(
                "Fine amount (€)", min_value=0.0, max_value=500.0, value=35.0, step=5.0,
                help="Median in dataset: ~94€",
            )
            points = st.number_input(
                "Penalty points", min_value=0, max_value=10, value=0, step=1,
                help="Range: 0–10",
            )
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Result")

        if not predict_clicked:
            st.info("Configure the case on the left and press **Predict**.")
            return

        if selected_k is None:
            st.error("Not enough activities selected.")
            return

        model = models.get((variant_key, selected_k))
        if model is None:
            st.error(f"Model not found for {variant_key.upper()}, k={selected_k}.")
            return

        fcols_path = f"outputs/models/feature_cols_outcome_{variant_key}_k{selected_k}.json"
        try:
            fcols = joblib.load(fcols_path)
        except FileNotFoundError:
            st.error(f"Feature columns not found: {fcols_path}")
            return

        # Build feature vector
        input_data = {col: 0 for col in fcols}
        for act in prefix_events:
            if act in input_data:
                input_data[act] = 1
        if "duration_so_far_days" in input_data:
            input_data["duration_so_far_days"] = duration_days
        if variant_key == "da":
            if "amount_sum_prefix" in input_data:
                input_data["amount_sum_prefix"] = amount
            if "points_sum_prefix" in input_data:
                input_data["points_sum_prefix"] = points

        X = pd.DataFrame([input_data], columns=fcols).fillna(0)
        prob = model.predict_proba(X)[0][1]

        # Display probability
        st.metric("Credit Collection Probability", f"{prob * 100:.1f}%")
        st.progress(float(prob))

        # Model info
        with st.expander("Model Details"):
            st.markdown(f"""
- **Model:** XGBoost
- **Variant:** {'Control-Flow only (binary activity indicators)' if variant_key == 'cf' else 'Data-Aware (activities + temporal + payload)'}
- **Prefix length (k):** {selected_k}
- **Features:** {len(fcols)} columns
- **Input activities:** {' → '.join(prefix_events)}
            """)

        # Per-instance SHAP explanation
        st.divider()
        st.subheader("Feature Contributions (this prediction)")
        try:
            import shap
            import matplotlib.pyplot as plt

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            elif shap_values.ndim == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]

            abs_sv = np.abs(sv)
            nonzero = np.where(abs_sv > 1e-8)[0]
            nonzero_sorted = nonzero[np.argsort(abs_sv[nonzero])[::-1]]
            top_idx = nonzero_sorted[:10]

            # Build labels showing feature name + its value
            x_row = X.iloc[0]
            labels = []
            for i in reversed(top_idx):
                val = x_row[fcols[i]]
                labels.append(f"{fcols[i]} = {val:.0f}" if val == int(val) else f"{fcols[i]} = {val:.2f}")

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#e74c3c" if sv[i] > 0 else "#3498db" for i in top_idx]
            ax.barh(
                labels,
                [sv[i] for i in reversed(top_idx)],
                color=[colors[j] for j in reversed(range(len(top_idx)))],
                edgecolor="black",
                linewidth=0.5,
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value (impact on collection probability)")
            ax.set_title("Top 10 features for this prediction")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.caption(
                "🔴 Red = increases collection risk · 🔵 Blue = decreases collection risk\n\n"
                "The value after `=` shows the feature's input (1 = activity observed, 0 = not observed). "
                "An absent activity (=0) can still contribute — its *absence* carries information."
            )
        except Exception as e:
            st.warning(f"SHAP computation failed: {e}")

        # Global importance
        with st.expander("Global Feature Importance (all test cases)"):
            shap_path = f"outputs/plots/shap_{variant_key}_k{selected_k}.png"
            if os.path.exists(shap_path):
                st.image(Image.open(shap_path), use_container_width=True)
            else:
                st.info("Run `python -m src.t6_interpretability` to generate.")
