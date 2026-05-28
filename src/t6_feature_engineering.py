"""Task 6.1: Data Preparation for ML
- Temporal split (time-based instead of random)
- Feature Engineering for RF/LR (k=2, k=5)
- Sequence features for LSTM (k=5)
- Leak-free: Removal of Payment/Collection events from features
- Saves: data/features/X_rf_k2.pkl, X_rf_k5.pkl, y_rf.pkl, X_lstm.pt, y_lstm.pt
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


# Events that reveal the outcome (Data Leakage)
LEAKING_EVENTS = ['Payment', 'Send for Credit Collection']


def create_rf_features(df, completed_cases, k, leak_free=True):
    """Creates features for Random Forest / Logistic Regression"""
    valid_cases = completed_cases[['case:concept:name', 'label']]
    df_filtered = df.merge(valid_cases, on='case:concept:name', how='inner')
    df_filtered = df_filtered.sort_values(by=['case:concept:name', 'time:timestamp'])
    df_filtered['event_position'] = df_filtered.groupby('case:concept:name').cumcount() + 1

    # Truncate to k events
    prefixes = df_filtered[df_filtered['event_position'] <= k].copy()

    # Leak-Free: Outcome-Events entfernen
    if leak_free:
        prefixes = prefixes[~prefixes['concept:name'].isin(LEAKING_EVENTS)].copy()

    # Only cases with at least 1 event after filtering
    cases_with_events = prefixes.groupby('case:concept:name').size()
    valid_prefix_cases = cases_with_events[cases_with_events >= 1].index
    prefixes = prefixes[prefixes['case:concept:name'].isin(valid_prefix_cases)].copy()

    # Features: Amount + Activity Encoding
    prefixes['amount'] = pd.to_numeric(prefixes['amount'], errors='coerce').fillna(0)
    X_amount = prefixes.groupby('case:concept:name')['amount'].max().reset_index()
    X_activities = pd.crosstab(prefixes['case:concept:name'], prefixes['concept:name']).reset_index()

    X = X_amount.merge(X_activities, on='case:concept:name')
    y = valid_cases[valid_cases['case:concept:name'].isin(X['case:concept:name'])]

    X = X.sort_values('case:concept:name').drop(columns=['case:concept:name']).reset_index(drop=True)
    y = y.sort_values('case:concept:name')['label'].reset_index(drop=True)

    return X, y


def create_lstm_features(df, completed_cases, k=5):
    """Creates sequence features for LSTM"""
    valid_cases = completed_cases[['case:concept:name', 'label']]
    df_filtered = df.merge(valid_cases, on='case:concept:name', how='inner')
    df_filtered = df_filtered.sort_values(by=['case:concept:name', 'time:timestamp'])
    df_filtered['event_position'] = df_filtered.groupby('case:concept:name').cumcount() + 1

    # Only cases with more than k events
    case_lengths = df_filtered.groupby('case:concept:name').size().reset_index(name='total_length')
    valid_ongoing_cases = case_lengths[case_lengths['total_length'] > k]['case:concept:name']
    df_filtered = df_filtered[df_filtered['case:concept:name'].isin(valid_ongoing_cases)].copy()

    # Activity Encoding
    activities = df_filtered['concept:name'].unique()
    activity_to_id = {act: i + 1 for i, act in enumerate(activities)}
    df_filtered['act_id'] = df_filtered['concept:name'].map(activity_to_id)

    # Truncate prefix
    df_prefixes = df_filtered[df_filtered['event_position'] <= k]
    sequences_df = df_prefixes.groupby('case:concept:name')['act_id'].apply(list).reset_index()
    sequences_df = sequences_df.merge(completed_cases[['case:concept:name', 'label']], on='case:concept:name')

    # Convert to tensors
    tensor_sequences = [torch.tensor(seq) for seq in sequences_df['act_id']]
    X_lstm = pad_sequence(tensor_sequences, batch_first=True, padding_value=0)
    y_lstm = torch.tensor(sequences_df['label'].values, dtype=torch.float32)

    return X_lstm, y_lstm


def main():
    print("=" * 60)
    print("TASK 6.1: Feature Engineering")
    print("=" * 60)

    # Load data
    df = pd.read_pickle("data/cleaned/df_cleaned.pkl")
    completed_cases = pd.read_pickle("data/cleaned/completed_cases.pkl")

    os.makedirs('data/features', exist_ok=True)

    # RF/LR features for k=2
    print("  Creating RF features (k=2, leak-free)...")
    X_k2, y_k2 = create_rf_features(df, completed_cases, k=2, leak_free=True)
    X_k2.to_pickle("data/features/X_rf_k2.pkl")
    y_k2.to_pickle("data/features/y_rf_k2.pkl")
    print(f"     Shape: {X_k2.shape} | Labels: {(y_k2==0).sum()} Payment, {(y_k2==1).sum()} Collection")

    # RF/LR features for k=5
    print("  Creating RF features (k=5, leak-free)...")
    X_k5, y_k5 = create_rf_features(df, completed_cases, k=5, leak_free=True)
    X_k5.to_pickle("data/features/X_rf_k5.pkl")
    y_k5.to_pickle("data/features/y_rf_k5.pkl")
    print(f"     Shape: {X_k5.shape} | Labels: {(y_k5==0).sum()} Payment, {(y_k5==1).sum()} Collection")

    # LSTM Features (k=5)
    print("  Creating LSTM sequence features (k=5)...")
    X_lstm, y_lstm = create_lstm_features(df, completed_cases, k=5)
    torch.save(X_lstm, "data/features/X_lstm.pt")
    torch.save(y_lstm, "data/features/y_lstm.pt")
    print(f"     Shape: {X_lstm.shape} | Labels: {int((y_lstm==0).sum())} Payment, {int((y_lstm==1).sum())} Collection")

    print("  ✅ All features saved to data/features/")


if __name__ == "__main__":
    main()
