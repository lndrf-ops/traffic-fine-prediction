"""Tab 4: Live Case Prediction."""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import shap
except ImportError:
    shap = None


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
    "Payment",
]

# Directly-follows relations observed in the real RTFM data.
# Used to constrain the trace builder to realistic sequences.
# "Send for Credit Collection" excluded — it's the outcome we predict.
VALID_SUCCESSORS = {
    "Create Fine": ["Send Fine", "Payment", "Insert Date Appeal to Prefecture", "Appeal to Judge"],
    "Send Fine": ["Insert Fine Notification", "Payment", "Insert Date Appeal to Prefecture", "Send Appeal to Prefecture", "Appeal to Judge"],
    "Insert Fine Notification": ["Add penalty", "Payment", "Insert Date Appeal to Prefecture", "Send Appeal to Prefecture", "Appeal to Judge", "Receive Result Appeal from Prefecture"],
    "Add penalty": ["Payment", "Insert Date Appeal to Prefecture", "Send Appeal to Prefecture", "Appeal to Judge", "Notify Result Appeal to Offender", "Receive Result Appeal from Prefecture"],
    "Insert Date Appeal to Prefecture": ["Send Fine", "Send Appeal to Prefecture", "Insert Fine Notification", "Add penalty", "Payment", "Receive Result Appeal from Prefecture", "Appeal to Judge"],
    "Send Appeal to Prefecture": ["Receive Result Appeal from Prefecture", "Payment", "Insert Fine Notification", "Add penalty", "Insert Date Appeal to Prefecture", "Notify Result Appeal to Offender", "Send Fine", "Appeal to Judge"],
    "Receive Result Appeal from Prefecture": ["Notify Result Appeal to Offender", "Payment", "Add penalty", "Appeal to Judge", "Insert Date Appeal to Prefecture", "Send Appeal to Prefecture"],
    "Notify Result Appeal to Offender": ["Payment", "Add penalty", "Appeal to Judge", "Send Appeal to Prefecture", "Receive Result Appeal from Prefecture"],
    "Appeal to Judge": ["Payment", "Send Fine", "Add penalty", "Insert Date Appeal to Prefecture", "Send Appeal to Prefecture", "Notify Result Appeal to Offender", "Receive Result Appeal from Prefecture"],
    "Payment": ["Payment", "Add penalty", "Send Fine", "Insert Fine Notification", "Insert Date Appeal to Prefecture", "Send Appeal to Prefecture", "Notify Result Appeal to Offender", "Receive Result Appeal from Prefecture", "Appeal to Judge"],
}

# Activities that require "Send Fine" to have occurred earlier in the trace
REQUIRES_SEND_FINE = {"Add penalty", "Insert Fine Notification", "Notify Result Appeal to Offender"}


def _get_valid_successors(trace: list[str]) -> list[str]:
    """Context-aware successors: filter based on what has occurred in the trace."""
    last = trace[-1]
    candidates = VALID_SUCCESSORS.get(last, [])
    seen = set(trace)
    # If Send Fine hasn't occurred, exclude activities that require it
    if "Send Fine" not in seen:
        candidates = [a for a in candidates if a not in REQUIRES_SEND_FINE]
    return candidates


# Always use model matching trace length (k=2 for 2 events, k=3 for 3-4, k=5 for 5+)
AVAILABLE_K = [2, 3, 5]


def _best_k(n_events: int) -> int | None:
    """Select the largest trained k that is <= n_events."""
    valid = [k for k in AVAILABLE_K if k <= n_events]
    return max(valid) if valid else None


def render(models: dict):
    st.header("Live Case Prediction")

    st.markdown(
        "Build a realistic case trace step by step. At each position, only activities "
        "that actually follow the previous one in the real data are available. "
        "The model predicts the probability of **credit collection** based on the observed pattern."
    )
    st.caption(
        "Successor constraints are derived from directly-follows relations in the RTFM event log. "
        "The model uses binary features (activity observed: yes/no), not the exact sequence order. "
        "Note: *Payment* events can occur mid-process without resolving the case."
    )

    col_in, col_out = st.columns([1, 2])

    with col_in:
        # Variant selection FIRST (stable, not affected by trace changes)
        variant = st.radio(
            "Feature variant",
            ["Control-Flow only", "Data-Aware (+ temporal, amount, …)"],
            key="variant_radio",
            help="CF uses only binary activity indicators. DA adds prefix length, duration, amount, points, and other case attributes.",
        )
        variant_key = "cf" if variant.startswith("Control") else "da"

        duration_days = 0
        amount = 35.0
        points = 0
        vehicle_class = "A"
        article = 157.0
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
            vehicle_class = st.selectbox(
                "Vehicle class",
                options=["A", "C", "M", "R"],
                index=0,
                key="vehicle_class_select",
                format_func=lambda x: {"A": "A – Autoveicoli (Cars)", "C": "C – Camion (Trucks)", "M": "M – Motoveicoli (Motorcycles)", "R": "R – Rimorchi (Trailers)"}[x],
                help="A = cars (97%), C = trucks, M = motorcycles, R = trailers",
            )
            article = st.selectbox(
                "Traffic article violated",
                options=[157, 7, 158, 142, 181, 180, 171, 80, 172, 146],
                index=0,
                key="article_select",
                format_func=lambda x: f"Art. {x}" + {157: " (speeding)", 7: " (red light)", 158: " (speeding minor)"}.get(x, ""),
                help="Top articles by frequency. Art. 157 = speeding (45% of cases)",
            )

        st.divider()
        st.subheader("Build Case Trace")

        @st.fragment
        def _trace_builder():
            # Sequential trace builder
            if "trace" not in st.session_state:
                st.session_state.trace = ["Create Fine"]

            # Display current trace
            trace = st.session_state.trace
            st.markdown("**Current trace:**", help="Model is selected based on trace length: 2 events → k=2, 3–4 → k=3, 5+ → k=5. Longer prefixes give more accurate predictions.")
            trace_str = " → ".join(f"`{a}`" for a in trace)
            st.markdown(trace_str)

            # Next activity selector (constrained to valid successors)
            last_activity = trace[-1]
            valid_next = _get_valid_successors(trace)

            if valid_next and len(trace) < 8:
                next_act = st.selectbox(
                    "Add next activity:",
                    options=valid_next,
                    key="next_activity_select",
                )
                col_add, col_reset = st.columns(2)
                with col_add:
                    if st.button("➕ Add", width="stretch"):
                        st.session_state.trace.append(next_act)
                        st.rerun(scope="fragment")
                with col_reset:
                    if st.button("🔄 Reset", width="stretch"):
                        st.session_state.trace = ["Create Fine"]
                        st.rerun(scope="fragment")
            else:
                if len(trace) >= 8:
                    st.info("Maximum prefix length reached (k=8).")
                else:
                    st.warning(f"No valid successors for '{last_activity}' (excluding end events).")
                if st.button("🔄 Reset trace", width="stretch"):
                    st.session_state.trace = ["Create Fine"]
                    st.rerun(scope="fragment")

        _trace_builder()

        prefix_events = st.session_state.get("trace", ["Create Fine"])
        n_events = len(prefix_events)
        selected_k = _best_k(n_events)

        st.divider()

        predict_clicked = st.button("Predict", type="primary", width="stretch")

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
            import json as _json
            with open(fcols_path) as _f:
                fcols = _json.load(_f)
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
            # One-hot encoded categoricals
            vc_col = f"vehicleClass_mode_{vehicle_class}"
            if vc_col in input_data:
                input_data[vc_col] = 1
            art_col = f"article_mode_{float(article)}"
            if art_col in input_data:
                input_data[art_col] = 1

        X = pd.DataFrame([input_data], columns=fcols).fillna(0)
        prob = model.predict_proba(X)[0][1]

        # Display probability
        st.metric("Credit Collection Probability", f"{prob * 100:.1f}%")
        st.progress(float(prob))

        # Prescriptive recommendation
        THRESHOLD_RED = 0.75
        THRESHOLD_YELLOW = 0.50
        if prob >= THRESHOLD_RED:
            st.error("🔴 **HIGH RISK** — Recommend immediate escalation (payment plan offer).")
        elif prob >= THRESHOLD_YELLOW:
            st.warning("🟡 **MEDIUM RISK** — Recommend proactive reminder.")
        else:
            st.success("🟢 **LOW RISK** — No intervention required. Case likely resolves via payment.")

        # Model info
        with st.expander("Model Details"):
            st.markdown(f"""
- **Model:** XGBoost
- **Variant:** {'Control-Flow only (binary activity indicators)' if variant_key == 'cf' else 'Data-Aware (activities + temporal + payload)'}
- **Prefix length (k):** {selected_k}
- **Features:** {len(fcols)} columns
- **Trace:** {' → '.join(prefix_events)}
- **Unique activities in trace:** {len(set(prefix_events))}
            """)

        # Per-instance SHAP explanation
        st.divider()
        st.subheader("Feature Contributions (this prediction)")
        try:
            import matplotlib.pyplot as plt

            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X.values)
            if isinstance(sv, list):
                sv = sv[1][0]  # class 1 (collection), first sample
            elif sv.ndim == 3:
                sv = sv[0, :, 1]
            elif sv.ndim == 2:
                sv = sv[0]

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
            ax.set_xlabel("SHAP value (impact on log-odds of collection)")
            ax.set_title("Top feature contributions for this prediction")
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
                st.image(Image.open(shap_path), width="stretch")
            else:
                st.info("Run `python -m src.t6_interpretability` to generate.")
