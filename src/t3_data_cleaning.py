"""Task 3: Data Cleaning
- Filtering and noise removal
- Duplicate removal
- Start/end event selection
- Data harmonization
- Case labeling (Payment vs. Credit Collection)
- Saves: data/cleaned/df_events.pkl -> data/cleaned/df_cleaned.pkl, completed_cases.pkl
"""

import os
import pandas as pd


def main():
    print("=" * 60)
    print("TASK 3: Data Cleaning")
    print("=" * 60)

    # 1. Daten laden
    df = pd.read_pickle("data/cleaned/df_events.pkl")
    print(f"  Loaded: {len(df):,} events, {df['case:concept:name'].nunique():,} cases")

    # 2. Sort by case and timestamp
    df = df.sort_values(by=['case:concept:name', 'time:timestamp']).reset_index(drop=True)

    # 3. Remove duplicates (same case, activity, timestamp)
    n_before = len(df)
    df = df.drop_duplicates(subset=['case:concept:name', 'concept:name', 'time:timestamp'])
    n_removed = n_before - len(df)
    print(f"  Duplicates removed: {n_removed:,}")

    # 4. Drop uninformative columns identified in t3 column profiling (fill rate < 1%, dummy values)
    cols_to_drop = [c for c in ['org:resource', 'matricola'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Dropped low-quality columns: {cols_to_drop}")

    # 5. Clean numeric columns
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)

    # 6. Case labeling (determine outcome)
    print("  Labeling cases (Payment=0, Credit Collection=1)...")
    # Group activities per case into a trace list. Store under `trace` to avoid
    # overwriting the original event column name `concept:name` in case downstream
    # code expects atomic event rows or uses `concept:name` as event column.
    cases = df.groupby('case:concept:name')['concept:name'].apply(list).reset_index()
    cases = cases.rename(columns={'concept:name': 'trace'})

    def determine_outcome(activity_list):
        # Credit Collection wins unconditionally — even if a payment occurred,
        # the process deviated and debt collectors were involved (label = 1).
        # Payment wins only if credit collection never happened (label = 0).
        # Open/running cases (neither event) are excluded from supervised learning.
        if 'Send for Credit Collection' in activity_list:
            return 1
        elif 'Payment' in activity_list:
            return 0
        else:
            return -1

    cases['label'] = cases['trace'].apply(determine_outcome)
    completed_cases = cases[cases['label'] != -1].copy()

    # 6b. Extract case-level attributes (from first event of each case)
    case_attrs = ['vehicleClass', 'article', 'points']
    available_attrs = [c for c in case_attrs if c in df.columns]
    if available_attrs:
        case_level = df.groupby('case:concept:name')[available_attrs].first().reset_index()
        completed_cases = completed_cases.merge(case_level, on='case:concept:name', how='left')
        print(f"  Case-level attributes added: {available_attrs}")

    n_payment = (completed_cases['label'] == 0).sum()
    n_collection = (completed_cases['label'] == 1).sum()
    n_incomplete = len(cases) - len(completed_cases)
    print(f"  Completed Cases:   {len(completed_cases):,}")
    print(f"    Payment (0):     {n_payment:,} ({n_payment / len(completed_cases) * 100:.1f}%)")
    print(f"    Collection (1):  {n_collection:,} ({n_collection / len(completed_cases) * 100:.1f}%)")
    print(f"  Incomplete Cases (no clear outcome, excluded): {n_incomplete:,}")

    # 7. Add event position
    df['event_position'] = df.groupby('case:concept:name').cumcount() + 1

    # 8. Speichern
    os.makedirs('data/cleaned', exist_ok=True)
    df.to_pickle("data/cleaned/df_cleaned.pkl")
    # Save completed_cases which now contains: case:concept:name, trace (list), label, + case-level attrs
    completed_cases.to_pickle("data/cleaned/completed_cases.pkl")

    print(f"  ✅ Cleaned data saved: data/cleaned/df_cleaned.pkl")
    print(f"  ✅ Labels saved: data/cleaned/completed_cases.pkl")
    return df, completed_cases


if __name__ == "__main__":
    main()
