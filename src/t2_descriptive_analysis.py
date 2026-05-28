"""Task 3: Descriptive Data Analysis (Data Exploration & Understanding)
- Column quality profiling (fill rates, min/max/median, unique values)
- Dataset properties (cases, events, activities, time range)
- Dotted Chart visualization
- Saves: outputs/plots/dotted_chart.png, outputs/reports/descriptive_analysis.json

Runs on df_events.pkl (raw, pre-cleaning) so all original columns are visible
and drop decisions in t2 can be justified from this analysis.
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    print("=" * 60)
    print("TASK 3: Descriptive Data Analysis")
    print("=" * 60)

    # Raw event log — no cleaning applied yet
    df = pd.read_pickle("data/cleaned/df_events.pkl")

    save_dir = "outputs/plots"
    report_dir = "outputs/reports"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # 1. Dataset properties
    n_cases = df["case:concept:name"].nunique()
    n_events = len(df)
    n_activities = df["concept:name"].nunique()
    activities = df["concept:name"].unique()

    print(f"  Cases:      {n_cases:,}")
    print(f"  Events:     {n_events:,}")
    print(f"  Activities: {n_activities}  {list(activities)}")
    print(f"  Time Range: {df['time:timestamp'].min()} – {df['time:timestamp'].max()}")

    events_per_case = df.groupby("case:concept:name").size()
    print(f"\n  Events per Case:")
    print(f"    Min: {events_per_case.min()} | Max: {events_per_case.max()} | "
          f"Mean: {events_per_case.mean():.1f} | Median: {events_per_case.median():.0f}")

    # 2. Column quality profiling
    LOW_FILL_THRESHOLD = 1.0
    print(f"\n  Column profiling ({len(df.columns)} columns):")
    print(f"\n  {'Column':<35} {'Dtype':<12} {'Fill%':>6}  {'Min':>12}  {'Median':>12}  {'Max':>12}  {'Unique':>8}")
    print(f"  {'-' * 95}")

    col_profiles = {}
    for col in df.columns:
        non_null = df[col].notna().sum()
        fill_pct = non_null / len(df) * 100
        series = df[col].dropna()
        if pd.api.types.is_numeric_dtype(df[col]):
            col_min = f"{series.min():.2f}" if len(series) else "—"
            col_med = f"{series.median():.2f}" if len(series) else "—"
            col_max = f"{series.max():.2f}" if len(series) else "—"
        else:
            col_min, col_med, col_max = "—", "—", "—"
        n_unique = df[col].nunique(dropna=True)
        print(f"  {col:<35} {str(df[col].dtype):<12} {fill_pct:>5.1f}%  "
              f"{col_min:>12}  {col_med:>12}  {col_max:>12}  {n_unique:>8,}")
        col_profiles[col] = {
            "dtype": str(df[col].dtype),
            "fill_rate_percent": round(fill_pct, 2),
            "non_null_count": int(non_null),
            "n_unique": int(n_unique),
            "min": float(series.min()) if pd.api.types.is_numeric_dtype(df[col]) and len(series) else None,
            "median": float(series.median()) if pd.api.types.is_numeric_dtype(df[col]) and len(series) else None,
            "max": float(series.max()) if pd.api.types.is_numeric_dtype(df[col]) and len(series) else None,
        }

    low_quality_cols = {
        col: profile for col, profile in col_profiles.items()
        if profile["fill_rate_percent"] < LOW_FILL_THRESHOLD or profile["n_unique"] <= 1
    }
    if low_quality_cols:
        print(f"\n  ⚠️  Low-quality columns (fill rate < {LOW_FILL_THRESHOLD}% or ≤1 unique value):")
        for col, profile in low_quality_cols.items():
            print(f"    - {col}: {profile['fill_rate_percent']:.2f}% filled, "
                  f"{profile['n_unique']} unique value(s) → recommended: DROP in t2")

    # 3. Save report
    report = {
        "dataset": {
            "total_cases": n_cases,
            "total_events": n_events,
            "distinct_activities": n_activities,
            "activity_names": list(activities),
            "time_range": {
                "start": str(df["time:timestamp"].min()),
                "end": str(df["time:timestamp"].max()),
            },
        },
        "events_per_case": {
            "min": int(events_per_case.min()),
            "max": int(events_per_case.max()),
            "mean": round(events_per_case.mean(), 2),
            "median": round(events_per_case.median(), 1),
            "std": round(events_per_case.std(), 2),
        },
        "columns": col_profiles,
        "low_quality_columns": {
            col: {
                "reason": (
                    "fill_rate < 1%" if p["fill_rate_percent"] < LOW_FILL_THRESHOLD
                    else "single unique value"
                )
            }
            for col, p in low_quality_cols.items()
        },
    }

    with open(f"{report_dir}/descriptive_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✅ Report saved: {report_dir}/descriptive_analysis.json")

    # 4. Dotted Chart
    print("\n  Generating Dotted Chart...")
    sample_cases = df["case:concept:name"].drop_duplicates().sample(500, random_state=42)
    df_sample = df[df["case:concept:name"].isin(sample_cases)].copy()
    df_sample = df_sample.sort_values(["case:concept:name", "time:timestamp"])

    case_start = df_sample.groupby("case:concept:name")["time:timestamp"].min().sort_values()
    case_index = {c: i for i, c in enumerate(case_start.index)}
    df_sample["case_idx"] = df_sample["case:concept:name"].map(case_index)

    fig, ax = plt.subplots(figsize=(14, 8))
    act_list = df_sample["concept:name"].unique()
    colors = plt.cm.tab10(range(len(act_list)))
    color_map = dict(zip(act_list, colors))

    for act in act_list:
        subset = df_sample[df_sample["concept:name"] == act]
        ax.scatter(subset["time:timestamp"], subset["case_idx"],
                   c=[color_map[act]], s=3, alpha=0.6, label=act)

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Case (sorted by start time)")
    ax.set_title("Dotted Chart (500 cases)")
    ax.legend(loc="upper left", fontsize=7, markerscale=4, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/dotted_chart.png", dpi=150)
    plt.close()
    print(f"  ✅ Dotted Chart saved: {save_dir}/dotted_chart.png")

    return df


if __name__ == "__main__":
    main()
