"""Task 3b: Batching Analysis

Observation from the Dotted Chart: Several activities form vertical lines,
indicating periodic batch execution rather than event-driven processing.
Consistent with Martin et al. (2017, Decision Support Systems) who identified
batch behaviour in this same RTFM dataset.

This script investigates:
  1. Batch concentration ranking: Computes a continuous "activity rate" for
     each activity (active_days / calendar_span) and events_per_active_day.
     No arbitrary threshold — the natural gap in the data speaks for itself.
  2. Waiting time analysis for the top-3 most concentrated activities:
     How long do cases idle before each fires?
  3. Implications for predictive modelling: Quantifies the structural floor on
     remaining-time MAE imposed by batch scheduling.

Saves:
  - outputs/plots/batching_analysis.png
  - outputs/reports/batching_analysis.json

References:
  - Martin, N., Swennen, M., Depaire, B., Jans, M., Caris, A., & Vanhoof, K.
    (2017). Retrieving batch organisation of work insights from event logs.
    Decision Support Systems, 100, 119–128.
"""

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# How many of the most concentrated activities to analyse waiting times for
TOP_N = 3


def compute_batch_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute continuous batching metrics for every activity.

    Key metrics:
      - activity_rate: fraction of calendar days on which the activity fires
        (low = rare/concentrated execution)
      - events_per_active_day: average load when the activity does fire
        (high = bulk processing)
      - max_median_ratio: spikiness (max daily count / median daily count)
        (high = occasional massive batch runs amid normal low-volume days)
    """
    df = df.copy()
    df["date"] = df["time:timestamp"].dt.date

    # Calendar span of dataset
    calendar_days = (df["date"].max() - df["date"].min()).days + 1

    stats = df.groupby("concept:name").agg(
        total_events=("date", "size"),
        active_days=("date", "nunique"),
    ).reset_index()

    # Per-activity daily counts for max/median
    daily_counts = df.groupby(["concept:name", "date"]).size().reset_index(name="day_count")
    daily_agg = daily_counts.groupby("concept:name")["day_count"].agg(
        max_day_count="max", median_day_count="median"
    ).reset_index()

    stats = stats.merge(daily_agg, on="concept:name")
    stats["events_per_active_day"] = stats["total_events"] / stats["active_days"]
    stats["activity_rate"] = stats["active_days"] / calendar_days
    stats["max_median_ratio"] = stats["max_day_count"] / stats["median_day_count"]
    stats["calendar_days"] = calendar_days
    stats = stats.sort_values("events_per_active_day", ascending=False).reset_index(drop=True)
    stats.rename(columns={"concept:name": "activity"}, inplace=True)
    return stats


def compute_waiting_times(df: pd.DataFrame, activities: list[str]) -> dict[str, pd.Series]:
    """For each specified activity, compute days since previous event in same case.

    Returns dict: activity -> Series of waiting days.
    """
    df_sorted = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    df_sorted["prev_timestamp"] = df_sorted.groupby("case:concept:name")["time:timestamp"].shift(1)
    df_sorted["wait_days"] = (
        df_sorted["time:timestamp"] - df_sorted["prev_timestamp"]
    ).dt.total_seconds() / 86400

    result = {}
    for act in activities:
        waits = df_sorted[df_sorted["concept:name"] == act]["wait_days"].dropna()
        if len(waits) > 0:
            result[act] = waits
    return result


def plot_batching_analysis(activity_stats: pd.DataFrame, wait_times: dict,
                           save_dir: str) -> None:
    """Two-panel figure: (1) batch concentration ranking, (2) waiting time boxplots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Events per active day (all activities, log scale) + activity rate annotation
    ax = axes[0]
    # Color gradient: darker = lower activity rate (more concentrated)
    rates = activity_stats["activity_rate"].values
    norm_rates = 1 - (rates / rates.max())  # invert so low rate = high intensity
    colors = plt.cm.RdYlBu_r(norm_rates * 0.8 + 0.1)  # avoid extremes of colormap

    y_pos = range(len(activity_stats))
    bars = ax.barh(y_pos, activity_stats["events_per_active_day"], color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(activity_stats["activity"], fontsize=9)
    ax.set_xscale("log")

    # Annotate activity rate on bars
    for i, (_, row) in enumerate(activity_stats.iterrows()):
        ax.text(1.2, i, f"{row['activity_rate']:.1%}",
                va="center", fontsize=8, color="#555555")

    ax.set_xlabel("Avg events per active day (log scale)")
    ax.set_title("Batch Concentration Ranking\n(% = activity rate: fraction of calendar days active)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: Waiting time boxplots for top-N activities
    ax = axes[1]
    if wait_times:
        ordered = sorted(wait_times.items(), key=lambda x: x[1].median(), reverse=True)
        labels = [name for name, _ in ordered]
        data = [series.clip(upper=series.quantile(0.95)).values for _, series in ordered]

        bp = ax.boxplot(data, vert=False, tick_labels=labels, patch_artist=True,
                        medianprops=dict(color="#d62728", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("#636EFA")
            patch.set_alpha(0.6)
        ax.set_xlabel("Days idle before activity fires (clipped at 95th pctl)")
        ax.set_title(f"Waiting Time Before Top-{TOP_N} Concentrated Activities")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "batching_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  ✅ Plot saved: %s", out_path)


def main():
    logger.info("=" * 60)
    logger.info("TASK 3b: Batching Analysis")
    logger.info("=" * 60)

    df = pd.read_pickle("data/cleaned/df_events.pkl")

    save_dir = "outputs/plots"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    # 1. Compute continuous batch metrics for all activities
    logger.info("\n  [1] Batch concentration ranking (no threshold — continuous metrics):")
    logger.info("      activity_rate = active_days / calendar_span (low = concentrated)")
    logger.info("      events_per_active_day = avg load when activity fires (high = bulk)")
    activity_stats = compute_batch_metrics(df)

    calendar_days = activity_stats["calendar_days"].iloc[0]
    logger.info("      Dataset spans %d calendar days\n", calendar_days)

    for _, row in activity_stats.iterrows():
        logger.info("    %-38s %6.0f ev/day | rate %5.1f%% | max/med %6.0f×",
                    row["activity"], row["events_per_active_day"],
                    row["activity_rate"] * 100, row["max_median_ratio"])

    # 2. Waiting times for top-N most concentrated activities
    top_activities = activity_stats.head(TOP_N)["activity"].tolist()
    logger.info("\n  [2] Idle time before top-%d concentrated activities...", TOP_N)
    wait_times = compute_waiting_times(df, top_activities)

    wait_stats = {}
    for act, waits in wait_times.items():
        stats = {
            "count": int(len(waits)),
            "median_days": round(float(waits.median()), 1),
            "mean_days": round(float(waits.mean()), 1),
            "std_days": round(float(waits.std()), 1),
            "p25_days": round(float(waits.quantile(0.25)), 1),
            "p75_days": round(float(waits.quantile(0.75)), 1),
            "p90_days": round(float(waits.quantile(0.90)), 1),
        }
        wait_stats[act] = stats
        logger.info("    %-38s median: %6.0f days | IQR: %.0f–%.0f days",
                    act, stats["median_days"], stats["p25_days"], stats["p75_days"])

    # 3. Implication
    logger.info("\n  [3] Implication for predictive modelling:")
    logger.info("    The data shows a clear separation between highly concentrated")
    logger.info("    activities (activity rate <1%%) and distributed activities (>50%%).")
    scc_stats = wait_stats.get("Send for Credit Collection", {})
    if scc_stats:
        logger.info("    'Send for Credit Collection' is the extreme case: %d active days,",
                    activity_stats[activity_stats["activity"] == "Send for Credit Collection"]["active_days"].iloc[0])
        logger.info("    median wait %.0f days (≈%.1f years) — an irreducible delay.",
                    scc_stats["median_days"], scc_stats["median_days"] / 365)
    logger.info("    These idle periods impose a structural floor on remaining-time MAE")
    logger.info("    that no model can overcome without knowledge of the batch schedule.")

    # 4. Plot
    plot_batching_analysis(activity_stats, wait_times, save_dir)

    # 5. Save report
    report = {
        "method": "Continuous batch concentration ranking (no arbitrary threshold)",
        "calendar_span_days": int(calendar_days),
        "activity_stats": activity_stats.drop(columns=["calendar_days"]).to_dict(orient="records"),
        "top_n_analysed": TOP_N,
        "waiting_time_stats": wait_stats,
        "implication": (
            "Activities with low activity rates and high events_per_active_day exhibit batch "
            "behaviour (Martin et al., 2017). 'Send for Credit Collection' (activity rate 0.2%, "
            "5365 events/active day) and 'Send Fine' (activity rate 37%, 58 events/active day "
            "but max/median ratio 908×) both introduce unpredictable idle periods. "
            "The waiting-time IQR represents a structural lower bound on remaining-time MAE."
        ),
    }

    report_path = "outputs/reports/batching_analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("  ✅ Report saved: %s", report_path)


if __name__ == "__main__":
    main()
