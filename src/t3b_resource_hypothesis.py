"""Task 3b: Resource Hypothesis Analysis
Hypothesis: NaN values in `org:resource` represent automated system steps (batch runs),
while filled values represent manual clerk actions.

Two tests:
  1. Throughput test (machine speed): events co-occurring in the exact same second
  2. Activity correlation: missing rate of `org:resource` per activity

Saves: outputs/plots/resource_missing_rate.png
"""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def throughput_test(df: pd.DataFrame) -> None:
    """Count events sharing the exact same timestamp, split by is_system."""
    # Floor to second precision for grouping
    df = df.copy()
    df["timestamp_sec"] = df["time:timestamp"].dt.floor("s")
    df["is_system"] = df["org:resource"].isna()

    events_per_second = (
        df.groupby(["is_system", "timestamp_sec"])
        .size()
        .reset_index(name="event_count")
    )

    for is_sys, label in [(True, "System (NaN resource)"), (False, "Human (filled resource)")]:
        subset = events_per_second[events_per_second["is_system"] == is_sys]["event_count"]
        # Only seconds with >1 event to measure batching; include all for mean
        logger.info("--- %s ---", label)
        logger.info(
            "  Max events in one second  : %d",
            subset.max(),
        )
        logger.info(
            "  Mean events per second    : %.2f",
            subset.mean(),
        )
        logger.info(
            "  Seconds with >1 event     : %d  (%.1f%% of all active seconds)",
            (subset > 1).sum(),
            (subset > 1).mean() * 100,
        )
        logger.info(
            "  Seconds with >10 events   : %d",
            (subset > 10).sum(),
        )


def activity_missing_rate_chart(df: pd.DataFrame, save_dir: str) -> None:
    """Bar chart of org:resource missing rate per activity, sorted descending."""
    missing_rate = (
        df.groupby("concept:name")["org:resource"]
        .apply(lambda s: s.isna().mean() * 100)
        .sort_values(ascending=True)  # ascending so highest ends up at top of hbar
        .reset_index()
    )
    missing_rate.columns = ["activity", "missing_rate_pct"]

    n = len(missing_rate)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.55)))

    colors = [
        "#d62728" if r >= 95 else "#ff7f0e" if r >= 50 else "#2ca02c"
        for r in missing_rate["missing_rate_pct"]
    ]

    bars = ax.barh(
        missing_rate["activity"],
        missing_rate["missing_rate_pct"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # Value labels at end of each bar
    for bar, val in zip(bars, missing_rate["missing_rate_pct"]):
        ax.text(
            min(val + 1, 97),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Missing rate of org:resource (%)", fontsize=11)
    ax.set_title(
        "org:resource Missing Rate per Activity\n"
        "(Red ≥ 95%: fully automated  |  Green < 50%: manual)",
        fontsize=12,
        pad=12,
    )
    ax.axvline(95, color="#d62728", linestyle="--", linewidth=1, alpha=0.6, label="95% threshold")
    ax.axvline(50, color="#ff7f0e", linestyle="--", linewidth=1, alpha=0.6, label="50% threshold")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "resource_missing_rate.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("  ✅ Chart saved: %s", out_path)


def main():
    logger.info("=" * 60)
    logger.info("TASK 3b: Resource Hypothesis Analysis")
    logger.info("=" * 60)

    df = pd.read_pickle("data/cleaned/df_events.pkl")

    save_dir = "outputs/plots"
    os.makedirs(save_dir, exist_ok=True)

    logger.info("\n[1] Throughput Test (machine speed via same-second co-occurrence)")
    throughput_test(df)

    logger.info("\n[2] Activity Correlation (org:resource missing rate per activity)")
    activity_missing_rate_chart(df, save_dir)


if __name__ == "__main__":
    main()
