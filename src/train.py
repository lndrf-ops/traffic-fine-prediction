import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Definition der LSTM Architektur (Muss beim Laden im App.py / evaluate.py bekannt sein)
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

def main():
    os.makedirs('models', exist_ok=True)
    
    # --- 1. RANDOM FOREST TRAINING ---
    print("Trainiere Random Forest...")
    X_rf = pd.read_pickle("data/processed/X_rf.pkl")
    y_rf = pd.read_pickle("data/processed/y_rf.pkl")
    
    X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(list(X_rf.columns), 'models/model_features.pkl')
    print("Random Forest gespeichert.")

    # --- 2. LSTM TRAINING ---
    print("\nTrainiere LSTM mit Class Weights...")
    X_seq_clean = torch.load("data/processed/X_lstm.pt", weights_only=True)
    y_seq_clean = torch.load("data/processed/y_lstm.pt", weights_only=True)
    
    dataset_clean = TensorDataset(X_seq_clean, y_seq_clean)
    train_size = int(0.8 * len(dataset_clean))
    test_size = len(dataset_clean) - train_size
    train_dataset, test_dataset = random_split(dataset_clean, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    num_payments = (y_seq_clean == 0).sum().item()
    num_collections = (y_seq_clean == 1).sum().item()
    weight_ratio = num_payments / num_collections
    pos_weight_tensor = torch.tensor([weight_ratio])
    
    # vocab_size auf 15 gesetzt als Puffer für verschiedene Log-Arten
    modell_lstm_final = ProcessLSTM(vocab_size=15, embedding_dim=16, hidden_dim=32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(modell_lstm_final.parameters(), lr=0.005)
    
    epochs = 5
    for epoch in range(epochs):
        modell_lstm_final.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = modell_lstm_final(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
    torch.save(modell_lstm_final.state_dict(), "models/lstm_model.pth")
    print("LSTM Modell gespeichert.")

if __name__ == "__main__":
    main()