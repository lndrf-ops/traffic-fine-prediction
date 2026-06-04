"""Task 6.2: Model Training

Trains all models in two variants (Control-Flow only / Data-Aware) for
two tasks (Outcome classification / Remaining Time regression), following
the model lineup in CLAUDE.md and the validation strategy in docs/validation_strategy.md.
"""

import os
import warnings

# --- MAC MULTIPROCESSING FIX ---
# Verhindert den "loky" Segmentation Fault auf Apple Silicon Macs
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
# -------------------------------

import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, XGBRegressor

PREFIX_LENGTHS = [2, 3, 5]  # longer prefixes (k≥5) show saturating performance
VARIANTS = ["cf", "da"]
SEED = 42

def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def load_split(variant: str, k: int):
    """Load train/val/test feature tables for a given variant and k."""
    df = pd.read_parquet(f"data/features/prefix_k{k}_{variant}.parquet")
    train = df[df["split"] == "train"].drop(columns=["split"])
    val = df[df["split"] == "val"].drop(columns=["split"])
    test = df[df["split"] == "test"].drop(columns=["split"])
    return train, val, test

def feature_cols(df: pd.DataFrame, task: str):
    """Return feature column names, excluding target columns."""
    exclude = {"label", "remaining_days", "split"}
    return [c for c in df.columns if c not in exclude]

# ---------------------------------------------------------------------------
# Classical models — Outcome (classification)
# ---------------------------------------------------------------------------

def train_outcome_classical(train: pd.DataFrame, val: pd.DataFrame, k: int, variant: str, save_dir: str):
    fcols = feature_cols(train, "outcome")
    X_train, y_train = train[fcols].values, train["label"].values
    X_val, y_val = val[fcols].values, val["label"].values

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / n_pos  # for XGBoost

    # Hyperparameters: defaults chosen following scikit-learn/XGBoost recommendations.
    # No grid search performed — with 4 activities (CF) the models saturate quickly;
    # tuning would yield marginal improvement at high computational cost.
    # NOTE: CF features at low k produce only 2–4 varying binary columns. All models
    # converge to the same case ranking (identical AUC-ROC) because the feature space
    # is too coarse to differentiate model families. This is expected and validates
    # the CF vs DA comparison: CF alone is insufficient for model differentiation.
    models = {
        "majority": DummyClassifier(strategy="most_frequent", random_state=SEED),
        "logreg": LogisticRegression(
            max_iter=2000, random_state=SEED, class_weight="balanced", C=1.0
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=1,  # n_jobs=1: macOS Apple Silicon fork safety
            class_weight="balanced"
        ),
        "xgb": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos,
            random_state=SEED,
            eval_metric="logloss",
            early_stopping_rounds=30,
            verbosity=0,
            n_jobs=1 # FIX: n_jobs=1
        ),
    }

    for name, model in models.items():
        if name == "xgb":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        path = f"{save_dir}/{name}_outcome_{variant}_k{k}.pkl"
        joblib.dump(model, path)

    with open(f"{save_dir}/feature_cols_outcome_{variant}_k{k}.json", "w") as f:
        json.dump(fcols, f)
    print(f"     Outcome classifiers saved ({variant.upper()}, k={k}): majority, logreg, rf, xgb")

# ---------------------------------------------------------------------------
# Classical models — Remaining Time (regression)
# ---------------------------------------------------------------------------

def train_remaining_classical(train: pd.DataFrame, val: pd.DataFrame, k: int, variant: str, save_dir: str):
    fcols = feature_cols(train, "remaining")
    X_train, y_train = train[fcols].values, train["remaining_days"].values
    X_val, y_val = val[fcols].values, val["remaining_days"].values

    models = {
        "mean": DummyRegressor(strategy="mean"),
        "linreg": LinearRegression(),
        "rf_reg": RandomForestRegressor(
            n_estimators=300, random_state=SEED, n_jobs=1 # FIX: n_jobs=1
        ),
        "xgb_reg": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=SEED,
            eval_metric="mae",
            early_stopping_rounds=30,
            verbosity=0,
            n_jobs=1 # FIX: n_jobs=1
        ),
    }

    for name, model in models.items():
        if name == "xgb_reg":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        path = f"{save_dir}/{name}_remaining_{variant}_k{k}.pkl"
        joblib.dump(model, path)

    with open(f"{save_dir}/feature_cols_remaining_{variant}_k{k}.json", "w") as f:
        json.dump(fcols, f)
    print(f"     Remaining-time regressors saved ({variant.upper()}, k={k}): mean, linreg, rf_reg, xgb_reg")

# ---------------------------------------------------------------------------
# LSTM — shared architecture for both tasks
# ---------------------------------------------------------------------------

from src.models import ProcessLSTM  # shared architecture (single source of truth)

def _load_lstm_data(variant: str, task: str, prefix_lengths: list):
    seqs_df = pd.read_parquet("data/features/sequences.parquet")
    seqs_df = seqs_df[seqs_df["prefix_k"].isin(prefix_lengths)].copy()

    target_col = "label" if task == "outcome" else "remaining_days"
    seqs_df = seqs_df.dropna(subset=[target_col])

    tensors = [torch.tensor(seq, dtype=torch.long) for seq in seqs_df["sequence"]]
    X = pad_sequence(tensors, batch_first=True, padding_value=0)
    y = torch.tensor(seqs_df[target_col].values, dtype=torch.float32)
    splits = seqs_df["split"].values
    return X, y, splits

def train_lstm(variant: str, task: str, vocab_size: int, save_dir: str):
    # One LSTM per (task, variant), pooled across all prefix lengths k ∈ {2,3,5}.
    # Rationale: sequences are very short (max 5 tokens from 9 activities) — training
    # separate models per k would fragment an already small feature space and reduce
    # training data per model. Pooling lets the LSTM learn general sequential patterns
    # while evaluation still reports per-k metrics for fair comparison with classical models.
    X, y, splits = _load_lstm_data(variant, task, PREFIX_LENGTHS)

    X_train = X[splits == "train"]
    y_train = y[splits == "train"]
    X_val = X[splits == "val"]
    y_val = y[splits == "val"]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = ProcessLSTM(
        vocab_size=vocab_size,
        embedding_dim=32,  
        hidden_dim=64,
    ).to(device)

    if task == "outcome":
        n_pos = (y_train == 1).sum().item()
        n_neg = (y_train == 0).sum().item()
        pos_weight = torch.tensor([n_neg / n_pos]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.SmoothL1Loss()  

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    max_epochs = 30
    early_stop_patience = 7  

    print(f"     Training LSTM ({task}, {variant.upper()}) on {device}...")
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter >= early_stop_patience:
            print(f"       Epoch {epoch:02d}/{max_epochs} — train: {train_loss:.4f}, val: {val_loss:.4f}")

        if patience_counter >= early_stop_patience:
            print(f"       Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    path = f"{save_dir}/lstm_{task}_{variant}.pth"
    torch.save(model.state_dict(), path)
    print(f"     LSTM {task} {variant.upper()} saved -> {path}")
    return model

def main():
    set_seeds()

    print("=" * 60)
    print("TASK 6.2: Model Training")
    print("=" * 60)

    save_dir = "outputs/models"
    os.makedirs(save_dir, exist_ok=True)

    with open("data/features/activity_vocab.json") as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"]

    # --- Classical models: one model per k, per variant, per task ---
    for variant in VARIANTS:
        print(f"\n  [{variant.upper()}] Classical models...")
        for k in PREFIX_LENGTHS:
            train, val, _ = load_split(variant, k)
            train_outcome_classical(train, val, k, variant, save_dir)
            train_remaining_classical(train, val, k, variant, save_dir)

    # --- LSTM: one model per variant per task (across all k) ---
    print("\n  LSTM models...")
    for variant in VARIANTS:
        for task in ["outcome", "remaining"]:
            train_lstm(variant, task, vocab_size, save_dir)

    print("\n  All models saved to outputs/models/")

if __name__ == "__main__":
    main()