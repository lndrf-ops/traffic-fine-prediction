"""Task 6.2: Model Training
- Random Forest training (k=2, k=5)
- Logistic Regression training (k=2, k=5)
- LSTM training (k=5)
- Saves: outputs/models/rf_k2.pkl, rf_k5.pkl, lr_k2.pkl, lr_k5.pkl, lstm.pth
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib


# LSTM Architecture
class ProcessLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(ProcessLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        final_out = out[:, -1, :]
        return self.fc(final_out).squeeze()


def train_rf_lr(X, y, k, save_dir):
    """Train Random Forest and Logistic Regression"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, f"{save_dir}/rf_k{k}.pkl")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    joblib.dump(lr, f"{save_dir}/lr_k{k}.pkl")

    # Feature names
    joblib.dump(list(X.columns), f"{save_dir}/features_k{k}.pkl")

    return rf, lr


def train_lstm(X_lstm, y_lstm, save_dir, epochs=5):
    """Train LSTM model"""
    dataset = TensorDataset(X_lstm, y_lstm)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Class weights
    num_payments = (y_lstm == 0).sum().item()
    num_collections = (y_lstm == 1).sum().item()
    pos_weight = torch.tensor([num_payments / num_collections])

    # Model
    model = ProcessLSTM(vocab_size=15, embedding_dim=16, hidden_dim=32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"     Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), f"{save_dir}/lstm.pth")
    return model


def main():
    print("=" * 60)
    print("TASK 6.2: Model Training")
    print("=" * 60)

    save_dir = 'outputs/models'
    os.makedirs(save_dir, exist_ok=True)

    # --- RF/LR k=2 ---
    print("  Training Random Forest & Logistic Regression (k=2)...")
    X_k2 = pd.read_pickle("data/features/X_rf_k2.pkl")
    y_k2 = pd.read_pickle("data/features/y_rf_k2.pkl")
    rf_k2, lr_k2 = train_rf_lr(X_k2, y_k2, k=2, save_dir=save_dir)
    print(f"     ✅ RF k=2 and LR k=2 saved")

    # --- RF/LR k=5 ---
    print("  Training Random Forest & Logistic Regression (k=5)...")
    X_k5 = pd.read_pickle("data/features/X_rf_k5.pkl")
    y_k5 = pd.read_pickle("data/features/y_rf_k5.pkl")
    rf_k5, lr_k5 = train_rf_lr(X_k5, y_k5, k=5, save_dir=save_dir)
    print(f"     ✅ RF k=5 and LR k=5 saved")

    # --- LSTM ---
    print("  Training LSTM (k=5)...")
    X_lstm = torch.load("data/features/X_lstm.pt", weights_only=True)
    y_lstm = torch.load("data/features/y_lstm.pt", weights_only=True)
    train_lstm(X_lstm, y_lstm, save_dir=save_dir, epochs=5)
    print(f"     ✅ LSTM saved")

    print("  ✅ All models saved to outputs/models/")


if __name__ == "__main__":
    main()
