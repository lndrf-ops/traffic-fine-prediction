Shared Model Definitions
- ProcessLSTM architecture (used by train and app)
"""

import torch
import torch.nn as nn


class ProcessLSTM(nn.Module):
    """LSTM model for process outcome prediction"""
    def __init__(self, vocab_size=15, embedding_dim=16, hidden_dim=32):
        super(ProcessLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        final_out = out[:, -1, :]
        return self.fc(final_out).squeeze()
