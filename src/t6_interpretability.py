"""Task 6.4: Interpretability
- SHAP Values (Post-hoc Explanation)
- Feature Importance
- Saves: outputs/plots/shap_k2.png, shap_k5.png
"""

import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def main():
    print("=" * 60)
    print("TASK 6.4: Interpretability (SHAP)")
    print("=" * 60)

    save_dir = 'outputs/plots'
    os.makedirs(save_dir, exist_ok=True)

    for k in [2, 5]:
        print(f"  Generating SHAP plot for RF (k={k})...")
        X = pd.read_pickle(f"data/features/X_rf_k{k}.pkl")
        y = pd.read_pickle(f"data/features/y_rf_k{k}.pkl")
        rf = joblib.load(f"outputs/models/rf_k{k}.pkl")

        _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_sample = X_test.sample(min(1000, len(X_test)), random_state=42)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)

        plt.figure()
        plt.title(f"SHAP Summary - Random Forest (k={k})")
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_sample, plot_type="dot", show=False)
        else:
            sv = shap_values[:, :, 1] if len(shap_values.shape) > 2 else shap_values
            shap.summary_plot(sv, X_sample, plot_type="dot", show=False)

        plt.savefig(f"{save_dir}/shap_k{k}.png", bbox_inches='tight')
        plt.close()
        print(f"     ✅ SHAP plot saved: {save_dir}/shap_k{k}.png")

    print("  ✅ Interpretability analysis complete")


if __name__ == "__main__":
    main()
