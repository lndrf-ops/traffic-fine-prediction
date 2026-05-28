"""Task 3: Descriptive Data Analysis (Data Exploration & Understanding)
- Dataset Properties (Cases, Events, Activities)
- Dotted Chart Analysis (Batching Behavior)
- Saves: outputs/plots/dotted_chart.png, outputs/reports/descriptive_analysis.json
"""

import os
import json
import pandas as pd
import pm4py
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("TASK 3: Descriptive Data Analysis")
    print("=" * 60)

    # 1. Daten laden
    df = pd.read_pickle("data/cleaned/df_cleaned.pkl")
    completed_cases = pd.read_pickle("data/cleaned/completed_cases.pkl")

    save_dir = 'outputs/plots'
    report_dir = 'outputs/reports'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # 2. Dataset Properties
    n_cases = df['case:concept:name'].nunique()
    n_events = len(df)
    n_activities = df['concept:name'].nunique()
    activities = df['concept:name'].unique()

    print(f"  Cases: {n_cases:,}")
    print(f"  Events: {n_events:,}")
    print(f"  Activities: {n_activities}")
    print(f"  Activity Names: {list(activities)}")
    print(f"  Completed Cases: {len(completed_cases):,}")
    print(f"  Time Range: {df['time:timestamp'].min()} – {df['time:timestamp'].max()}")

    # 2b. Column overview
    print(f"\n  Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"  Data Types:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"    - {col}: {df[col].dtype} ({non_null:,} non-null)")

    # 2c. Events per case statistics
    events_per_case = df.groupby('case:concept:name').size()
    print(f"\n  Events per Case:")
    print(f"    Min: {events_per_case.min()} | Max: {events_per_case.max()} | "
          f"Mean: {events_per_case.mean():.1f} | Median: {events_per_case.median():.0f}")

    # 2d. Save descriptive analysis report
    report = {
        "dataset": {
            "total_cases": n_cases,
            "total_events": n_events,
            "distinct_activities": n_activities,
            "activity_names": list(df['concept:name'].unique()),
            "completed_cases": len(completed_cases),
            "time_range": {
                "start": str(df['time:timestamp'].min()),
                "end": str(df['time:timestamp'].max())
            }
        },
        "columns": {
            col: {
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "fill_rate_percent": round(df[col].notna().sum() / len(df) * 100, 2)
            }
            for col in df.columns
        },
        "events_per_case": {
            "min": int(events_per_case.min()),
            "max": int(events_per_case.max()),
            "mean": round(events_per_case.mean(), 2),
            "median": round(events_per_case.median(), 1),
            "std": round(events_per_case.std(), 2)
        },
        "label_distribution": {
            "payment": int((completed_cases['label'] == 0).sum()),
            "collection": int((completed_cases['label'] == 1).sum()),
            "incomplete": n_cases - len(completed_cases)
        }
    }

    with open(f"{report_dir}/descriptive_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✅ Report saved: {report_dir}/descriptive_analysis.json")

    # 3. Dotted Chart (Batching Analysis)
    print("\n  Generating Dotted Chart (Batching Analysis)...")
    import matplotlib.pyplot as plt

    sample_cases = df['case:concept:name'].drop_duplicates().sample(500, random_state=42)
    df_sample = df[df['case:concept:name'].isin(sample_cases)].copy()
    df_sample = df_sample.sort_values(['case:concept:name', 'time:timestamp'])

    # Assign numeric case index sorted by first event time
    case_start = df_sample.groupby('case:concept:name')['time:timestamp'].min().sort_values()
    case_index = {c: i for i, c in enumerate(case_start.index)}
    df_sample['case_idx'] = df_sample['case:concept:name'].map(case_index)

    fig, ax = plt.subplots(figsize=(14, 8))
    activities = df_sample['concept:name'].unique()
    colors = plt.cm.tab10(range(len(activities)))
    color_map = dict(zip(activities, colors))

    for act in activities:
        subset = df_sample[df_sample['concept:name'] == act]
        ax.scatter(subset['time:timestamp'], subset['case_idx'],
                   c=[color_map[act]], s=3, alpha=0.6, label=act)

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Case (sorted by start time)")
    ax.set_title("Dotted Chart (500 cases)")
    ax.legend(loc='upper left', fontsize=7, markerscale=4, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/dotted_chart.png", dpi=150)
    plt.close()
    print(f"  ✅ Dotted Chart saved: {save_dir}/dotted_chart.png")

    return df


if __name__ == "__main__":
    main()
