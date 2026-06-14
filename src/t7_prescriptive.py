"""Task 7: Prescriptive Process Analytics

Threshold-based action policy using XGBoost outcome probabilities:
  - p >= 0.75  →  RED:    escalate immediately (payment plan offer)
  - p in [0.50, 0.75)  →  YELLOW: manual reminder / proactive contact
  - p < 0.50  →  GREEN:  no intervention required

Saves:
  - outputs/reports/prescriptive_recommendations.csv
  - outputs/plots/prescriptive_risk_tiers.png
"""

import os

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THRESHOLD_RED = 0.75
THRESHOLD_YELLOW = 0.50

# Cost-benefit assumptions (€, illustrative)
COST_ESCALATION = 15.0
COST_REMINDER = 5.0
BENEFIT_PREVENTED_COLLECTION = 120.0


def load_test_data(variant: str = "da", k: int = 5) -> tuple[np.ndarray, np.ndarray, list]:
    df = pd.read_parquet(f"data/features/prefix_k{k}_{variant}.parquet")
    test = df[df["split"] == "test"].drop(columns=["split"])
    fcols = [c for c in test.columns if c not in {"label", "remaining_days"}]
    return test[fcols].values, test["label"].values, fcols


def compute_expected_gain(prob: float, action: str) -> float:
    """Expected monetary gain of taking action vs. doing nothing.

    Gain = P(collection) * benefit_avoided - cost_of_action

    Args:
        prob: Predicted probability of credit collection (label=1).
        action: 'escalate', 'reminder', or 'none'.

    Returns:
        Expected net gain in €.
    """
    if action == "escalate":
        return prob * BENEFIT_PREVENTED_COLLECTION - COST_ESCALATION
    if action == "reminder":
        # Partial effect: reminders are less effective than direct escalation.
        # Assumption: ~50% conversion rate vs. full escalation, based on typical
        # debt-collection literature (e.g., early-stage nudges vs. formal demands).
        return prob * BENEFIT_PREVENTED_COLLECTION * 0.5 - COST_REMINDER  # partial effect
    return 0.0


def assign_policy(prob: float) -> tuple[str, str]:
    """Assign risk tier and recommended action based on probability."""
    if prob >= THRESHOLD_RED:
        return "RED", "escalate"
    if prob >= THRESHOLD_YELLOW:
        return "YELLOW", "reminder"
    return "GREEN", "none"


def main():
    print("=" * 60)
    print("BONUS: Prescriptive Process Analytics")
    print("=" * 60)

    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    model_path = "outputs/models/xgb_outcome_da_k5.pkl"
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        print("  Run 'python run_pipeline.py' first.")
        return

    model = joblib.load(model_path)
    X_test, y_test, fcols = load_test_data(variant="da", k=5)

    # Use the full test set for statistically robust prescriptive evaluation
    X_sample = X_test
    y_sample = y_test
    idx = np.arange(len(X_test))

    proba = model.predict_proba(X_sample)[:, 1]

    rows = []
    for i, (prob, true_label) in enumerate(zip(proba, y_sample)):
        tier, action = assign_policy(prob)
        gain = compute_expected_gain(prob, action)
        rows.append(
            {
                "case_index": idx[i],
                "predicted_risk": round(float(prob), 4),
                "true_label": int(true_label),
                "policy_tier": tier,
                "recommended_action": action,
                "expected_gain_eur": round(gain, 2),
            }
        )

    df_rec = pd.DataFrame(rows).sort_values("predicted_risk", ascending=False)
    out_csv = "outputs/reports/prescriptive_recommendations.csv"
    df_rec.to_csv(out_csv, index=False)
    print(f"  Recommendations saved: {out_csv}")

    # Summary statistics
    tier_counts = df_rec["policy_tier"].value_counts()
    total_gain = df_rec["expected_gain_eur"].sum()
    print(f"\n  Policy Tier Summary (n={len(df_rec)}):")
    for tier in ["RED", "YELLOW", "GREEN"]:
        n = tier_counts.get(tier, 0)
        print(f"    {tier:<8} {n:>4} cases")
    print(f"  Total expected net gain: €{total_gain:,.2f}")

    # --- Risk tier scatter plot ---
    color_map = {"RED": "#EF553B", "YELLOW": "#FFA15A", "GREEN": "#00CC96"}
    colors = df_rec["policy_tier"].map(color_map)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        range(len(df_rec)),
        df_rec["predicted_risk"],
        c=colors,
        s=40,
        alpha=0.8,
        edgecolors="none",
    )
    ax.axhline(THRESHOLD_RED, color="#EF553B", linestyle="--", linewidth=1.2,
               label=f"RED threshold ({THRESHOLD_RED:.0%})")
    ax.axhline(THRESHOLD_YELLOW, color="#FFA15A", linestyle="--", linewidth=1.2,
               label=f"YELLOW threshold ({THRESHOLD_YELLOW:.0%})")

    patches = [
        mpatches.Patch(color=color_map["RED"], label=f"RED — escalate ({tier_counts.get('RED', 0)})"),
        mpatches.Patch(color=color_map["YELLOW"], label=f"YELLOW — reminder ({tier_counts.get('YELLOW', 0)})"),
        mpatches.Patch(color=color_map["GREEN"], label=f"GREEN — no action ({tier_counts.get('GREEN', 0)})"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    ax.set_xlabel("Case rank (sorted by predicted risk, descending)")
    ax.set_ylabel("P(Credit Collection)")
    ax.set_title("Prescriptive Policy: Risk Tiers for Active Cases (XGBoost DA, k=5)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = "outputs/plots/prescriptive_risk_tiers.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
