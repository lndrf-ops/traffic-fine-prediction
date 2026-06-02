"""Validation helper: check event-log granularity and trace lengths.

Usage:
    python scripts/validate_event_log.py

Prints summary statistics and sample traces/cases helpful to debug preprocessing.
"""
import pandas as pd


def main():
    completed_cases = pd.read_pickle("data/cleaned/completed_cases.pkl")
    df = pd.read_pickle("data/cleaned/df_cleaned.pkl")

    print("Loaded:")
    print(f"  df_cleaned events: {len(df):,}, cases: {df['case:concept:name'].nunique():,}")
    print(f"  completed_cases rows: {len(completed_cases):,}")

    # If completed_cases stores a 'trace' column (list of activities), inspect it
    if 'trace' in completed_cases.columns:
        traces = completed_cases['trace']
        is_list = traces.apply(lambda x: isinstance(x, list))
        print(f"  completed_cases.trace contains lists: {is_list.any()}")
        if is_list.any():
            lens = traces[is_list].apply(len)
            print("  trace length stats:")
            print(lens.describe())
            print("  fraction single-event cases (trace length == 1):", (lens == 1).mean())
            print("  sample traces (first 10):")
            for i, t in enumerate(traces.head(10).tolist(), 1):
                print(f"    {i}: {t}")
    else:
        # Otherwise, infer trace lengths from df_cleaned event positions
        if 'event_position' in df.columns:
            lens = df.groupby('case:concept:name')['event_position'].max()
            print("  inferred trace length stats from df_cleaned.event_position:")
            print(lens.describe())
            print("  fraction single-event cases:", (lens == 1).mean())
        else:
            print("  No trace column and no event_position available to infer trace lengths.")

    # Quick check: how many cases have only 1 event in df_cleaned
    counts = df.groupby('case:concept:name').size()
    single_event_frac = (counts == 1).mean()
    print(f"  fraction of cases with single event in df_cleaned: {single_event_frac:.3f}")


if __name__ == '__main__':
    main()
