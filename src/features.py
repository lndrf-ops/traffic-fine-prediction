import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

def main():
    print("Starte Feature Engineering...")
    
    # 1. Daten laden
    df = pd.read_pickle("data/processed/df_raw.pkl")
    completed_cases = pd.read_pickle("data/processed/completed_cases.pkl")
    
    # --- A) FEATURES FÜR RANDOM FOREST (k=2) ---
    print("Erstelle Features für Random Forest (k=2)...")
    valid_cases = completed_cases[['case:concept:name', 'label']]
    df_filtered = df.merge(valid_cases, on='case:concept:name', how='inner')
    df_filtered = df_filtered.sort_values(by=['case:concept:name', 'time:timestamp'])
    df_filtered['event_position'] = df_filtered.groupby('case:concept:name').cumcount() + 1
    
    k_rf = 2
    prefixes_rf = df_filtered[df_filtered['event_position'] <= k_rf].copy()
    cases_with_min_k = prefixes_rf.groupby('case:concept:name').size()
    valid_prefix_cases = cases_with_min_k[cases_with_min_k == k_rf].index
    prefixes_rf = prefixes_rf[prefixes_rf['case:concept:name'].isin(valid_prefix_cases)].copy()
    
    prefixes_rf['amount'] = pd.to_numeric(prefixes_rf['amount'], errors='coerce').fillna(0)
    X_amount = prefixes_rf.groupby('case:concept:name')['amount'].max().reset_index()
    X_activities = pd.crosstab(prefixes_rf['case:concept:name'], prefixes_rf['concept:name']).reset_index()
    
    X_rf = X_amount.merge(X_activities, on='case:concept:name')
    y_rf = valid_cases[valid_cases['case:concept:name'].isin(X_rf['case:concept:name'])]
    
    X_rf = X_rf.sort_values('case:concept:name').drop(columns=['case:concept:name'])
    y_rf = y_rf.sort_values('case:concept:name')['label']
    
    X_rf.to_pickle("data/processed/X_rf.pkl")
    y_rf.to_pickle("data/processed/y_rf.pkl")
    
    # --- B) FEATURES FÜR LSTM (k=5, Leak-Free) ---
    print("Erstelle Sequenz-Features für LSTM (k=5, Leak-Free)...")
    k_lstm = 5
    case_lengths = df_filtered.groupby('case:concept:name').size().reset_index(name='total_length')
    valid_ongoing_cases = case_lengths[case_lengths['total_length'] > k_lstm]['case:concept:name']
    
    df_filtered_leakage = df_filtered[df_filtered['case:concept:name'].isin(valid_ongoing_cases)].copy()
    
    activities = df_filtered_leakage['concept:name'].unique()
    activity_to_id = {act: i+1 for i, act in enumerate(activities)}
    df_filtered_leakage['act_id'] = df_filtered_leakage['concept:name'].map(activity_to_id)
    
    df_prefixes_clean = df_filtered_leakage[df_filtered_leakage['event_position'] <= k_lstm]
    sequences_clean_df = df_prefixes_clean.groupby('case:concept:name')['act_id'].apply(list).reset_index()
    sequences_clean_df = sequences_clean_df.merge(completed_cases[['case:concept:name', 'label']], on='case:concept:name')
    
    tensor_sequences_clean = [torch.tensor(seq) for seq in sequences_clean_df['act_id']]
    X_seq_lstm = pad_sequence(tensor_sequences_clean, batch_first=True, padding_value=0)
    y_seq_lstm = torch.tensor(sequences_clean_df['label'].values, dtype=torch.float32)
    
    torch.save(X_seq_lstm, "data/processed/X_lstm.pt")
    torch.save(y_seq_lstm, "data/processed/y_lstm.pt")
    
    print("Alle Features erfolgreich generiert und gespeichert!")

if __name__ == "__main__":
    main()