"""Task 1: Data Collection and Event Log Construction
- Process Identification
- Building the Event Log for the Road Traffic Fine Management Process
- Saves: data/raw/ -> data/cleaned/df_events.pkl
"""

import os
import pm4py
import pandas as pd


def main():
    print("=" * 60)
    print("TASK 1: Data Loading & Event Log Construction")
    print("=" * 60)

    # 1. Path to XES file
    xes_path = "data/raw/Road_Traffic_Fine_Management_Process.xes.gz"
    if not os.path.exists(xes_path):
        xes_path = "data/raw/Road_Traffic_Fine_Management_Process.xes"
        if not os.path.exists(xes_path):
            raise FileNotFoundError(f"File {xes_path} not found!")

    # 2. Load event log
    print(f"  Reading event log from: {xes_path}")
    event_log = pm4py.read_xes(xes_path)
    df = pm4py.convert_to_dataframe(event_log)

    # 3. Basic statistics
    n_cases = df['case:concept:name'].nunique()
    n_events = len(df)
    n_activities = df['concept:name'].nunique()
    print(f"  Cases: {n_cases:,} | Events: {n_events:,} | Activities: {n_activities}")

    # 4. Save raw DataFrame
    os.makedirs('data/cleaned', exist_ok=True)
    df.to_pickle("data/cleaned/df_events.pkl")

    print(f"  ✅ Event log saved: data/cleaned/df_events.pkl")
    return df


if __name__ == "__main__":
    main()
