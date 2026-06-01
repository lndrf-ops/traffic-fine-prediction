"""Task 6.4: Interpretability via SHAP

Generates SHAP summary plots for the XGBoost outcome classifiers
across all prefix lengths k ∈ {2, 3, 5, 8} and both variants (CF, DA).

SHAP TreeExplainer is used because XGBoost is a tree ensemble — exact Shapley
values can be computed efficiently without approximation (Lundberg et al., 2020).

Saves: outputs/plots/shap_{variant}_k{k}.png
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PREFIX_LENGTHS = [2, 3, 5, 8]
VARIANTS = ["cf", "da"]


def shap_summary(variant: str, k: int, model_dir: str, plot_dir: str, n_sample: int = 200):
    model_path = f"{model_dir}/xgb_outcome_{variant}_k{k}.pkl"
    if not os.path.exists(model_path):
        print(f"     Skipping {variant} k={k} — model not found")
        return

    df = pd.read_parquet(f"data/features/prefix_k{k}_{variant}.parquet")
    test = df[df["split"] == "test"].drop(columns=["split"])
    fcols = [c for c in test.columns if c not in {"label", "remaining_days"}]
    X_test = test[fcols]

    model = joblib.load(model_path)
    X_sample = X_test.sample(min(n_sample, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # XGBoost binary: shap_values may be 2D (single output) or list [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values  # XGBoost returns single array for binary classification

    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_sample, plot_type="dot", show=False, max_display=15)
    plt.title(f"SHAP Feature Importance — XGBoost Outcome ({variant.upper()}, k={k})", fontsize=12)
    plt.tight_layout()
    out_path = f"{plot_dir}/shap_{variant}_k{k}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {out_path}")


def main():
    print("=" * 60)
    print("TASK 6.4: Interpretability (SHAP)")
    print("=" * 60)

    model_dir = "outputs/models"
    plot_dir = "outputs/plots"
    os.makedirs(plot_dir, exist_ok=True)

    for variant in VARIANTS:
        print(f"  [{variant.upper()}] Generating SHAP plots...")
        for k in PREFIX_LENGTHS:
            shap_summary(variant, k, model_dir, plot_dir)

    print("  Interpretability complete.")


if __name__ == "__main__":
    main()
