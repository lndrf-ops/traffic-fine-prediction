"""Task 6.3: Model Evaluation

Evaluates all trained models on the held-out test split using the metrics
defined in docs/validation_strategy.md:

  Outcome (classification):  AUC-ROC (primary), F1 positive class, Brier Score
  Remaining Time (regression): MAE in days (primary), RMSE

Models evaluated:
  - majority / mean (baselines)
  - logreg / linreg
  - rf / rf_reg
  - xgb / xgb_reg
  - LSTM (outcome + remaining, CF + DA)

All models loaded from outputs/models/; test split from data/features/split_indices.json.
Never re-trains — test set is only touched here, after all training decisions are final.

Saves:
  outputs/reports/evaluation_results.json   — full metrics table (consumed by reporting)
  outputs/plots/eval_outcome_auc.png        — AUC-ROC comparison across models × k
  outputs/plots/eval_remaining_mae.png      — MAE comparison across models × k
"""

import os

# Must be set before any native library (XGBoost, sklearn) spawns threads on macOS Apple Silicon
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

PREFIX_LENGTHS = [2, 3, 5]  # longer prefixes (k≥5) show saturating performance
VARIANTS = ["cf", "da"]
CLASSICAL_OUTCOME = ["majority", "logreg", "rf", "xgb"]
CLASSICAL_REMAINING = ["mean", "linreg", "rf_reg", "xgb_reg"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split(variant: str, k: int, split: str):
    df = pd.read_parquet(f"data/features/prefix_k{k}_{variant}.parquet")
    subset = df[df["split"] == split].drop(columns=["split"])
    fcols = [c for c in subset.columns if c not in {"label", "remaining_days"}]
    return subset, fcols


def load_test_split(variant: str, k: int):
    return load_split(variant, k, "test")


def outcome_metrics(y_true, y_pred, y_proba) -> dict:
    """Per-class and overall metrics for binary outcome prediction."""
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Collection class (positive = 1)
        "f1_collection": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_collection": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_collection": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        # Payment class (negative = 0)
        "f1_payment": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_payment": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_payment": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
    }


def remaining_metrics(y_true, y_pred) -> dict:
    """MAE and RMSE in days."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae_days": float(mae), "rmse_days": float(rmse)}


# ---------------------------------------------------------------------------
# Classical model evaluation
# ---------------------------------------------------------------------------

def eval_classical_outcome(variant: str, k: int, model_dir: str) -> list[dict]:
    test, fcols = load_test_split(variant, k)
    X_test = test[fcols].values
    y_test = test["label"].values
    rows = []
    for name in CLASSICAL_OUTCOME:
        path = f"{model_dir}/{name}_outcome_{variant}_k{k}.pkl"
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        # DummyClassifier has predict_proba; all others too
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "task": "outcome",
            "model": name,
            "variant": variant,
            "k": k,
            **outcome_metrics(y_test, y_pred, y_proba),
        })
    return rows


def eval_classical_remaining(variant: str, k: int, model_dir: str) -> list[dict]:
    test, fcols = load_test_split(variant, k)
    X_test = test[fcols].values
    y_test = test["remaining_days"].values
    rows = []
    for name in CLASSICAL_REMAINING:
        path = f"{model_dir}/{name}_remaining_{variant}_k{k}.pkl"
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        rows.append({
            "task": "remaining",
            "model": name,
            "variant": variant,
            "k": k,
            **remaining_metrics(y_test, y_pred),
        })
    return rows


# ---------------------------------------------------------------------------
# LSTM evaluation
# ---------------------------------------------------------------------------

from src.models import ProcessLSTM  # shared architecture (single source of truth)


def eval_lstm(variant: str, task: str, vocab_size: int, model_dir: str) -> list[dict]:
    path = f"{model_dir}/lstm_{task}_{variant}.pth"
    if not os.path.exists(path):
        return []

    seqs_df = pd.read_parquet("data/features/sequences.parquet")
    test_df = seqs_df[seqs_df["split"] == "test"].copy()
    target_col = "label" if task == "outcome" else "remaining_days"
    test_df = test_df.dropna(subset=[target_col])

    tensors = [torch.tensor(seq, dtype=torch.long) for seq in test_df["sequence"]]
    X = pad_sequence(tensors, batch_first=True, padding_value=0)
    y = torch.tensor(test_df[target_col].values, dtype=torch.float32)

    model = ProcessLSTM(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()

    loader = DataLoader(TensorDataset(X, y), batch_size=512, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for bx, _ in loader:
            preds.append(model(bx).numpy())
    y_pred_raw = np.concatenate(preds)
    y_true = y.numpy()

    rows = []
    for k in PREFIX_LENGTHS:
        mask = test_df["prefix_k"].values == k
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred_raw[mask]
        if task == "outcome":
            y_proba = 1 / (1 + np.exp(-yp))  # sigmoid
            y_class = (y_proba >= 0.5).astype(int)
            rows.append({
                "task": "outcome",
                "model": "lstm",
                "variant": variant,
                "k": k,
                **outcome_metrics(yt, y_class, y_proba),
            })
        else:
            rows.append({
                "task": "remaining",
                "model": "lstm",
                "variant": variant,
                "k": k,
                **remaining_metrics(yt, yp),
            })
    return rows


# ---------------------------------------------------------------------------
# Overfitting assessment
# ---------------------------------------------------------------------------

def _gap_verdict_outcome(gap: float) -> str:
    """Classify AUC-ROC train/test gap.
    
    Negative gap (test > train) can occur with temporal splits due to
    class distribution shift between time periods — not a bug.
    """
    if gap < -0.02:
        return "distribution_shift"
    if gap < 0.02:
        return "healthy"
    if gap < 0.05:
        return "mild"
    return "overfit"


def _gap_verdict_remaining(rel_gap: float) -> str:
    """Classify relative MAE train/test gap (test_mae / train_mae - 1)."""
    if rel_gap < -0.05:
        return "distribution_shift"
    if rel_gap < 0.05:
        return "healthy"
    if rel_gap < 0.15:
        return "mild"
    return "overfit"


def assess_overfitting(model_dir: str, vocab_size: int) -> list[dict]:
    """Compute train vs. test metric gap for every model × variant × k.

    Returns one dict per combination with fields:
        task, model, variant, k,
        train_<metric>, test_<metric>, gap, verdict
    """
    rows = []

    for variant in VARIANTS:
        for k in PREFIX_LENGTHS:
            train_data, _ = load_split(variant, k, "train")
            test_data, fcols = load_split(variant, k, "test")

            X_train = train_data[fcols].values
            y_train_outcome = train_data["label"].values
            y_train_remaining = train_data["remaining_days"].values

            X_test = test_data[fcols].values
            y_test_outcome = test_data["label"].values
            y_test_remaining = test_data["remaining_days"].values

            # --- Classical outcome models ---
            for name in CLASSICAL_OUTCOME:
                path = f"{model_dir}/{name}_outcome_{variant}_k{k}.pkl"
                if not os.path.exists(path):
                    continue
                model = joblib.load(path)
                train_proba = model.predict_proba(X_train)[:, 1]
                test_proba = model.predict_proba(X_test)[:, 1]
                train_auc = float(roc_auc_score(y_train_outcome, train_proba))
                test_auc = float(roc_auc_score(y_test_outcome, test_proba))
                gap = train_auc - test_auc
                rows.append({
                    "task": "outcome",
                    "model": name,
                    "variant": variant,
                    "k": k,
                    "train_auc_roc": round(train_auc, 4),
                    "test_auc_roc": round(test_auc, 4),
                    "gap": round(gap, 4),
                    "verdict": _gap_verdict_outcome(gap),
                })

            # --- Classical remaining-time models ---
            for name in CLASSICAL_REMAINING:
                path = f"{model_dir}/{name}_remaining_{variant}_k{k}.pkl"
                if not os.path.exists(path):
                    continue
                model = joblib.load(path)
                train_mae = float(mean_absolute_error(y_train_remaining, model.predict(X_train)))
                test_mae = float(mean_absolute_error(y_test_remaining, model.predict(X_test)))
                rel_gap = (test_mae / train_mae - 1) if train_mae > 0 else 0.0
                rows.append({
                    "task": "remaining",
                    "model": name,
                    "variant": variant,
                    "k": k,
                    "train_mae_days": round(train_mae, 2),
                    "test_mae_days": round(test_mae, 2),
                    "rel_gap": round(rel_gap, 4),
                    "verdict": _gap_verdict_remaining(rel_gap),
                })

    # --- LSTM: evaluate on train and test sequences ---
    seqs_df = pd.read_parquet("data/features/sequences.parquet")
    lstm_metrics: dict = defaultdict(dict)
    for variant in VARIANTS:
        for task in ["outcome", "remaining"]:
            path = f"{model_dir}/lstm_{task}_{variant}.pth"
            if not os.path.exists(path):
                continue

            target_col = "label" if task == "outcome" else "remaining_days"
            model_lstm = ProcessLSTM(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64)
            model_lstm.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model_lstm.eval()

            for split_name in ["train", "test"]:
                split_df = seqs_df[seqs_df["split"] == split_name].dropna(subset=[target_col])
                tensors = [torch.tensor(seq, dtype=torch.long) for seq in split_df["sequence"]]
                X = pad_sequence(tensors, batch_first=True, padding_value=0)
                y = split_df[target_col].values

                loader = DataLoader(TensorDataset(X), batch_size=512, shuffle=False, num_workers=0)
                preds = []
                with torch.no_grad():
                    for (bx,) in loader:
                        preds.append(model_lstm(bx).numpy())
                y_pred_raw = np.concatenate(preds)

                for k in PREFIX_LENGTHS:
                    mask = split_df["prefix_k"].values == k
                    if mask.sum() == 0:
                        continue
                    yt = y[mask]
                    yp = y_pred_raw[mask]
                    key = ("lstm", variant, k)
                    if task == "outcome":
                        y_proba = 1 / (1 + np.exp(-yp))
                        auc = float(roc_auc_score(yt, y_proba))
                        lstm_metrics[(key, task)][split_name] = auc
                    else:
                        mae = float(mean_absolute_error(yt, yp))
                        lstm_metrics[(key, task)][split_name] = mae

    # Convert collected LSTM train/test metrics into gap records
    for (key, task), splits in lstm_metrics.items():
        _, variant, k = key
        train_val = splits.get("train")
        test_val = splits.get("test")
        if train_val is None or test_val is None:
            continue
        if task == "outcome":
            gap = train_val - test_val
            rows.append({
                "task": "outcome",
                "model": "lstm",
                "variant": variant,
                "k": k,
                "train_auc_roc": round(train_val, 4),
                "test_auc_roc": round(test_val, 4),
                "gap": round(gap, 4),
                "verdict": _gap_verdict_outcome(gap),
            })
        else:
            rel_gap = (test_val / train_val - 1) if train_val > 0 else 0.0
            rows.append({
                "task": "remaining",
                "model": "lstm",
                "variant": variant,
                "k": k,
                "train_mae_days": round(train_val, 2),
                "test_mae_days": round(test_val, 2),
                "rel_gap": round(rel_gap, 4),
                "verdict": _gap_verdict_remaining(rel_gap),
            })

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_outcome(results: list[dict], save_path: str):
    """AUC-ROC bar chart grouped by k, one panel per variant."""
    df = pd.DataFrame([r for r in results if r["task"] == "outcome"])
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Outcome Prediction — AUC-ROC by Model and Prefix Length", fontweight="bold")
    models = df["model"].unique()
    x = np.arange(len(PREFIX_LENGTHS))
    width = 0.8 / len(models)
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for ax, variant in zip(axes, VARIANTS):
        sub = df[df["variant"] == variant]
        for i, (model, color) in enumerate(zip(models, colors)):
            aucs = [
                sub[(sub["model"] == model) & (sub["k"] == k)]["auc_roc"].values
                for k in PREFIX_LENGTHS
            ]
            aucs = [v[0] if len(v) else np.nan for v in aucs]
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, aucs, width, label=model, color=color, alpha=0.85)

        ax.set_title(f"Variant: {variant.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in PREFIX_LENGTHS])
        ax.set_xlabel("Prefix Length")
        ax.set_ylabel("AUC-ROC")
        ax.set_ylim(0.5, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_remaining(results: list[dict], save_path: str):
    """MAE bar chart grouped by k, one panel per variant."""
    df = pd.DataFrame([r for r in results if r["task"] == "remaining"])
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Remaining Time Prediction — MAE (days) by Model and Prefix Length", fontweight="bold")
    models = df["model"].unique()
    x = np.arange(len(PREFIX_LENGTHS))
    width = 0.8 / len(models)
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for ax, variant in zip(axes, VARIANTS):
        sub = df[df["variant"] == variant]
        for i, (model, color) in enumerate(zip(models, colors)):
            maes = [
                sub[(sub["model"] == model) & (sub["k"] == k)]["mae_days"].values
                for k in PREFIX_LENGTHS
            ]
            maes = [v[0] if len(v) else np.nan for v in maes]
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, maes, width, label=model, color=color, alpha=0.85)

        ax.set_title(f"Variant: {variant.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in PREFIX_LENGTHS])
        ax.set_xlabel("Prefix Length")
        ax.set_ylabel("MAE (days)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(model_dir: str):
    """Generate confusion matrices for XGBoost at k=5 (both variants)."""
    for variant in VARIANTS:
        k = 5
        path = f"{model_dir}/xgb_outcome_{variant}_k{k}.pkl"
        if not os.path.exists(path):
            continue
        test, fcols = load_test_split(variant, k)
        X_test = test[fcols].values
        y_test = test["label"].values

        model = joblib.load(path)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Collection", "Collection"])
        disp.plot(ax=ax, cmap="Blues", values_format=",d")
        ax.set_title(f"Confusion Matrix — XGBoost ({variant.upper()}, k={k})")
        plt.tight_layout()
        plt.savefig(f"outputs/plots/confusion_matrix_{variant}_k{k}.png", dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("TASK 6.3: Model Evaluation")
    print("=" * 60)

    model_dir = "outputs/models"
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    with open("data/features/activity_vocab.json") as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"]

    all_results = []

    # Classical models
    for variant in VARIANTS:
        for k in PREFIX_LENGTHS:
            all_results += eval_classical_outcome(variant, k, model_dir)
            all_results += eval_classical_remaining(variant, k, model_dir)

    # LSTM
    for variant in VARIANTS:
        for task in ["outcome", "remaining"]:
            all_results += eval_lstm(variant, task, vocab_size, model_dir)

    # Print summary tables
    df = pd.DataFrame(all_results)

    print("\n  OUTCOME PREDICTION (AUC-ROC on test set)")
    print(f"  {'Model':<12} {'Variant':<6} {'k=2':>7} {'k=3':>7} {'k=5':>7}")
    print(f"  {'─'*42}")
    for model in CLASSICAL_OUTCOME + ["lstm"]:
        for variant in VARIANTS:
            sub = df[(df["task"] == "outcome") & (df["model"] == model) & (df["variant"] == variant)]
            aucs = [sub[sub["k"] == k]["auc_roc"].values for k in PREFIX_LENGTHS]
            aucs_str = "  ".join(f"{v[0]:.4f}" if len(v) else "  —   " for v in aucs)
            print(f"  {model:<12} {variant.upper():<6} {aucs_str}")

    print("\n  REMAINING TIME (MAE days on test set)")
    print(f"  {'Model':<12} {'Variant':<6} {'k=2':>7} {'k=3':>7} {'k=5':>7}")
    print(f"  {'─'*42}")
    for model in CLASSICAL_REMAINING + ["lstm"]:
        for variant in VARIANTS:
            sub = df[(df["task"] == "remaining") & (df["model"] == model) & (df["variant"] == variant)]
            maes = [sub[sub["k"] == k]["mae_days"].values for k in PREFIX_LENGTHS]
            maes_str = "  ".join(f"{v[0]:>7.1f}" if len(v) else "    —  " for v in maes)
            print(f"  {model:<12} {variant.upper():<6} {maes_str}")

    # Save JSON
    with open("outputs/reports/evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n  Evaluation results saved: outputs/reports/evaluation_results.json")

    # Plots
    plot_outcome(all_results, "outputs/plots/eval_outcome_auc.png")
    plot_remaining(all_results, "outputs/plots/eval_remaining_mae.png")
    plot_confusion_matrices(model_dir)
    print("  Plots saved: outputs/plots/eval_outcome_auc.png, eval_remaining_mae.png, confusion_matrix_*.png")

    # Overfitting assessment
    print("\n  Computing overfitting assessment (train vs. test gaps)...")
    of_rows = assess_overfitting(model_dir, vocab_size)
    with open("outputs/reports/overfitting_assessment.json", "w") as f:
        json.dump(of_rows, f, indent=2)

    # Print summary
    df_of = pd.DataFrame(of_rows)
    verdict_counts = df_of["verdict"].value_counts()
    print(f"  Verdicts: ok={verdict_counts.get('ok', 0)}, "
          f"mild={verdict_counts.get('mild', 0)}, "
          f"overfit={verdict_counts.get('overfit', 0)}")
    overfit_models = df_of[df_of["verdict"] == "overfit"][["task", "model", "variant", "k"]]
    if not overfit_models.empty:
        print("  Overfit cases:")
        for _, row in overfit_models.iterrows():
            print(f"    {row['model']:12} {row['variant'].upper()} k={row['k']} ({row['task']})")
    print("  Overfitting assessment saved: outputs/reports/overfitting_assessment.json")


if __name__ == "__main__":
    main()
