"""Shared Model Definitions — single source of truth for neural network architectures.

Used by: t6_train.py, t6_evaluate.py
"""

import torch
import torch.nn as nn


class ProcessLSTM(nn.Module):
    """LSTM for process prediction (outcome classification or remaining time regression).

    Args:
        vocab_size: Number of activity types + padding token
        embedding_dim: Dimension of activity embeddings
        hidden_dim: LSTM hidden state dimension
        output_dim: 1 for binary classification (logit) or regression
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 16, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        out = self.dropout(h_n[-1])
        return self.fc(out).squeeze(-1)
