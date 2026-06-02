"""Task 6.1: Feature Engineering for Predictive Process Monitoring

Builds prefix-based feature sets following Teinemaa et al. (2019):
- Prefix lengths k ∈ {2, 3, 5, 8}
- Two feature variants per k: Control-Flow only (CF) and Data-Aware (DA)
- Two prediction targets: outcome (classification) and remaining time (regression)
- Temporal split: 64% train / 16% val / 20% test (sorted by case start timestamp)
- Debiased split: cases crossing train/test boundary are dropped (Teinemaa 2019, §4)
- All aggregates are prefix-bounded — no future leakage

Saves:
  data/features/prefix_k{k}_{cf|da}.parquet  — classical ML features per k
  data/features/sequences.parquet            — LSTM sequences (all k, padded)
  data/features/split_indices.json           — train/val/test case IDs
"""

import json
import os

import numpy as np
import pandas as pd

# Outcome-revealing activities — must never appear in the prefix (leakage guardrail #1)
OUTCOME_ACTIVITIES = {"Payment", "Send for Credit Collection"}

# Prefix lengths to evaluate (Teinemaa 2019, §3.2)
# Longer prefixes (k≥5) show saturating performance.
PREFIX_LENGTHS = [2, 3, 5]

# Leakage-safe payload columns for the Data-Aware variant.
#
# Dropped attributes and reasons:
#   totalPaymentAmount — cumulative payment, only known at case end (future leakage)
#   paymentAmount      — individual payment amount, leaks outcome (only exists if case = Payment)
#   expense            — notification cost, accrues over time (future leakage)
#   notificationType   — 99.8% = "P", near-constant; redundant with activity "Insert Fine Notification"
#   dismissal          — 98.5% = "NIL", near-constant, negligible predictive value
#   org:resource       — 73% missing, 148 unique IDs; outcome variation ±12pp is weak and
#                        confounded with case attributes (amount, article) already in the model
#   lifecycle:transition — single value ("complete"), zero information
#   lastSent           — 86% missing, 3 unique values, redundant with activity sequence
#   matricola          — 99.9% missing, single unique value
#
# Numeric: summed over prefix events. Categorical: most-frequent value (mode) in prefix.
DA_PAYLOAD_COLS_NUMERIC = ["amount", "points"]
DA_PAYLOAD_COLS_CATEGORICAL = ["vehicleClass", "article"]
DA_PAYLOAD_COLS = DA_PAYLOAD_COLS_NUMERIC + DA_PAYLOAD_COLS_CATEGORICAL

# Top-10 articles cover 98.5% of cases; group the remaining 56 as "Other"
# to reduce dimensionality (66 → 11 one-hot columns).
# Distribution: Art. 157 (speeding, 45%), Art. 7 (red light, 29%),
# Art. 158 (minor speeding, 18%) → top 3 alone = 92%.
ARTICLE_TOP_N = [157.0, 7.0, 158.0, 142.0, 181.0, 180.0, 171.0, 80.0, 172.0, 146.0]


def split_temporal(df_events: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.2):
    """Temporal split on case start timestamps.

    Args:
        df_events: Full cleaned event log.
        train_ratio: Fraction of earliest cases forming the train pool.
        val_ratio: Fraction of the train pool held out for validation (latest cases).

    Returns:
        (train_cases, val_cases, test_cases): sets of case IDs.
    """
    case_start = (
        df_events.groupby("case:concept:name")["time:timestamp"]
        .min()
        .sort_values()
        .reset_index()
    )
    case_start.columns = ["case:concept:name", "start_time"]

    n = len(case_start)
    train_pool_end = int(n * train_ratio)

    train_pool = case_start.iloc[:train_pool_end]
    test_pool = case_start.iloc[train_pool_end:]

    val_start_idx = int(len(train_pool) * (1 - val_ratio))
    train_cases = set(train_pool.iloc[:val_start_idx]["case:concept:name"])
    val_cases = set(train_pool.iloc[val_start_idx:]["case:concept:name"])
    test_cases = set(test_pool["case:concept:name"])

    return train_cases, val_cases, test_cases


def debias_split(df_events: pd.DataFrame, train_cases: set, val_cases: set, test_cases: set):
    """Drop cases whose [start, end] interval crosses any split boundary.

    Cases that span the train/val or val/test boundary share temporal context
    with both sides and are dropped to prevent leakage (Teinemaa 2019, §4).
    """
    # Compute per-case time range
    case_range = df_events.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
    case_range.columns = ["start", "end"]

    # Boundary timestamps
    train_end = df_events[df_events["case:concept:name"].isin(train_cases)]["time:timestamp"].max()
    val_end = df_events[df_events["case:concept:name"].isin(val_cases)]["time:timestamp"].max()

    # A case crosses if its end is after the boundary of its own split
    def is_crossing(case_id, start, end):
        if case_id in train_cases:
            return end > train_end
        if case_id in val_cases:
            return end > val_end
        return False

    crossed = {
        case_id
        for case_id, row in case_range.iterrows()
        if is_crossing(case_id, row["start"], row["end"])
    }

    train_cases -= crossed
    val_cases -= crossed
    # test cases are the latest — nothing to cross into
    return train_cases, val_cases, test_cases, len(crossed)


def compute_remaining_time(df_events: pd.DataFrame) -> pd.Series:
    """Compute per-case end timestamp (tz stripped to date-level for safe arithmetic).

    For a prefix of length k, remaining time = case_end - timestamp of event k.
    We store case_end here; the per-prefix subtraction happens in build_prefixes.
    All timestamps are date-only (00:00:00) per the RTFM dataset — tz is dropped
    after groupby so subsequent arithmetic never mixes tz-aware and tz-naive values.
    """
    case_end = df_events.groupby("case:concept:name")["time:timestamp"].max()
    # Strip timezone: dates are all 00:00:00 UTC so tz carries no information
    case_end = case_end.dt.tz_localize(None)
    return case_end


def build_prefixes(
    df_events: pd.DataFrame,
    completed_cases: pd.DataFrame,
    k: int,
    variant: str,
    case_end_times: pd.Series,
) -> pd.DataFrame:
    """Build a prefix feature table for prefix length k.

    Args:
        df_events: Cleaned event log (all events).
        completed_cases: DataFrame with columns [case:concept:name, label].
        k: Prefix length.
        variant: 'cf' (Control-Flow only) or 'da' (Data-Aware).
        case_end_times: Series case_id -> case end timestamp.

    Returns:
        DataFrame with one row per case, columns: features + 'label' + 'remaining_days' + 'split'.
    """
    valid_ids = set(completed_cases["case:concept:name"])
    label_map = completed_cases.set_index("case:concept:name")["label"].to_dict()

    df = df_events[df_events["case:concept:name"].isin(valid_ids)].copy()
    df = df.sort_values(["case:concept:name", "time:timestamp"])

    # Assign within-case event position
    df["_pos"] = df.groupby("case:concept:name").cumcount() + 1

    # Truncate at position k, then remove outcome-revealing activities (guardrail #1)
    df_prefix = df[df["_pos"] <= k].copy()
    df_prefix = df_prefix[~df_prefix["concept:name"].isin(OUTCOME_ACTIVITIES)].copy()

    # Drop cases where no events remain after leakage removal
    keep = df_prefix.groupby("case:concept:name").size()
    keep = keep[keep >= 1].index
    df_prefix = df_prefix[df_prefix["case:concept:name"].isin(keep)].copy()

    # --- Control-Flow features ---
    # Activity counts (one-hot aggregated over prefix)
    act_counts = (
        pd.crosstab(df_prefix["case:concept:name"], df_prefix["concept:name"])
        .reset_index()
    )

    features = act_counts

    # --- Data-Aware features: add temporal + payload on top of CF ---
    if variant == "da":
        # Duration so far in days (prefix-aware, not full case)
        prefix_start = df_prefix.groupby("case:concept:name")["time:timestamp"].min()
        prefix_last = df_prefix.groupby("case:concept:name")["time:timestamp"].max()
        duration_so_far = ((prefix_last - prefix_start).dt.total_seconds() / 86400).rename(
            "duration_so_far_days"
        )
        features = features.merge(duration_so_far.reset_index(), on="case:concept:name")

        for col in DA_PAYLOAD_COLS_NUMERIC:
            if col not in df_prefix.columns:
                continue
            # Sum over prefix — safe because we only use prefix events
            agg = df_prefix.groupby("case:concept:name")[col].sum().rename(f"{col}_sum_prefix")
            features = features.merge(agg.reset_index(), on="case:concept:name", how="left")

        for col in DA_PAYLOAD_COLS_CATEGORICAL:
            if col not in df_prefix.columns:
                continue
            # Most-frequent value in prefix (mode); categorical, not numeric
            agg = (
                df_prefix.groupby("case:concept:name")[col]
                .agg(lambda s: s.mode().iloc[0] if not s.isna().all() else np.nan)
                .rename(f"{col}_mode")
            )
            features = features.merge(agg.reset_index(), on="case:concept:name", how="left")

        # One-hot encode categorical DA columns
        # Group rare articles into "Other" before encoding
        if "article_mode" in features.columns:
            features["article_mode"] = features["article_mode"].apply(
                lambda x: x if x in ARTICLE_TOP_N else "Other"
            )
        cat_cols = [c for c in features.columns if c.endswith("_mode")]
        features = pd.get_dummies(features, columns=cat_cols, dummy_na=False)

    # --- Targets ---
    features["label"] = features["case:concept:name"].map(label_map)

    # Remaining time: case_end - timestamp of the k-th (or last available) prefix event
    # Strip tz from prefix timestamps (case_end_times is already tz-naive via compute_remaining_time)
    last_prefix_ts = df_prefix.groupby("case:concept:name")["time:timestamp"].max().dt.tz_localize(None)
    case_ids = features["case:concept:name"].values
    end_ts = pd.Series([case_end_times.get(cid, pd.NaT) for cid in case_ids], index=case_ids)
    last_ts = last_prefix_ts.reindex(case_ids).values
    remaining = (end_ts.values - last_ts) / np.timedelta64(1, "D")
    features["remaining_days"] = remaining

    features = features.drop(columns=["case:concept:name"])
    return features


def main():
    print("=" * 60)
    print("TASK 6.1: Feature Engineering")
    print("=" * 60)

    df = pd.read_pickle("data/cleaned/df_cleaned.pkl")
    completed_cases = pd.read_pickle("data/cleaned/completed_cases.pkl")

    os.makedirs("data/features", exist_ok=True)

    # 1. Temporal split
    print("  1. Computing temporal split...")
    train_cases, val_cases, test_cases = split_temporal(df)
    train_cases, val_cases, test_cases, n_crossed = debias_split(df, train_cases, val_cases, test_cases)

    print(f"     Train: {len(train_cases):,} cases")
    print(f"     Val:   {len(val_cases):,} cases")
    print(f"     Test:  {len(test_cases):,} cases")
    print(f"     Dropped (boundary-crossing): {n_crossed:,} cases")

    # Only retain completed cases that survived the split
    all_split_cases = train_cases | val_cases | test_cases
    completed_cases = completed_cases[
        completed_cases["case:concept:name"].isin(all_split_cases)
    ].copy()

    # Build a split column for each case
    def split_label(cid):
        if cid in train_cases:
            return "train"
        if cid in val_cases:
            return "val"
        return "test"

    split_map = {cid: split_label(cid) for cid in all_split_cases}

    # Save split indices
    split_indices = {
        "train": sorted(train_cases),
        "val": sorted(val_cases),
        "test": sorted(test_cases),
        "n_dropped_boundary": n_crossed,
    }
    with open("data/features/split_indices.json", "w") as f:
        json.dump(split_indices, f)
    print("     Split indices saved: data/features/split_indices.json")

    # 2. Case end times (for remaining time target)
    case_end_times = compute_remaining_time(df)

    # 3. Build prefix feature tables for k ∈ {2, 3, 5, 8}, variants CF and DA
    print(f"  2. Building prefix features for k ∈ {PREFIX_LENGTHS}, variants: cf, da...")
    for k in PREFIX_LENGTHS:
        for variant in ["cf", "da"]:
            features = build_prefixes(df, completed_cases, k, variant, case_end_times)

            # Attach split column
            # Re-merge case IDs from df to attach split — we need to recover them
            # Build features with case ID retained temporarily
            valid_ids = set(completed_cases["case:concept:name"])
            label_map = completed_cases.set_index("case:concept:name")["label"].to_dict()

            df_tmp = df[df["case:concept:name"].isin(valid_ids)].copy()
            df_tmp = df_tmp.sort_values(["case:concept:name", "time:timestamp"])
            df_tmp["_pos"] = df_tmp.groupby("case:concept:name").cumcount() + 1
            df_prefix_tmp = df_tmp[df_tmp["_pos"] <= k].copy()
            df_prefix_tmp = df_prefix_tmp[
                ~df_prefix_tmp["concept:name"].isin(OUTCOME_ACTIVITIES)
            ].copy()
            keep = df_prefix_tmp.groupby("case:concept:name").size()
            keep = keep[keep >= 1].index
            surviving_cases = list(keep)

            features["split"] = [split_map.get(cid, "unknown") for cid in surviving_cases]

            out_path = f"data/features/prefix_k{k}_{variant}.parquet"
            features.to_parquet(out_path, index=False)

            n_train = (features["split"] == "train").sum()
            n_val = (features["split"] == "val").sum()
            n_test = (features["split"] == "test").sum()
            n_cols = features.shape[1]
            print(
                f"     k={k} {variant.upper()}: {len(features):,} cases, {n_cols} features "
                f"(train={n_train:,} val={n_val:,} test={n_test:,}) -> {out_path}"
            )

    # 4. Sequence features for LSTM (all cases, variable length up to max k)
    # LSTM uses a single model over all prefix lengths — sequences padded in t6_train.py
    print("  3. Building LSTM sequence features...")
    _build_lstm_sequences(df, completed_cases, split_map, case_end_times)

    print("  Feature engineering complete.")


def _build_lstm_sequences(
    df: pd.DataFrame,
    completed_cases: pd.DataFrame,
    split_map: dict,
    case_end_times: pd.Series,
):
    """Build variable-length activity sequences for LSTM training.

    One row per (case, prefix_length) pair for k ∈ PREFIX_LENGTHS.
    Sequences stored as lists; padding handled in t6_train.py.
    Saves: data/features/sequences.parquet
    """
    valid_ids = set(completed_cases["case:concept:name"])
    label_map = completed_cases.set_index("case:concept:name")["label"].to_dict()

    df_s = df[df["case:concept:name"].isin(valid_ids)].copy()
    df_s = df_s.sort_values(["case:concept:name", "time:timestamp"])
    df_s["_pos"] = df_s.groupby("case:concept:name").cumcount() + 1

    # Remove outcome events globally (they can't be in any prefix)
    df_s = df_s[~df_s["concept:name"].isin(OUTCOME_ACTIVITIES)].copy()

    # Encode activities as integers (1-indexed; 0 reserved for padding)
    all_activities = sorted(df_s["concept:name"].unique())
    act_to_id = {a: i + 1 for i, a in enumerate(all_activities)}

    # Precompute case_end as tz-naive for subtraction
    case_end_naive = case_end_times  # already tz-naive from compute_remaining_time()

    rows = []
    for k in PREFIX_LENGTHS:
        df_k = df_s[df_s["_pos"] <= k].copy()
        df_k["act_id"] = df_k["concept:name"].map(act_to_id)

        # Vectorized: sequence and last timestamp per case
        seqs = df_k.groupby("case:concept:name")["act_id"].apply(list)
        last_ts_k = df_k.groupby("case:concept:name")["time:timestamp"].max().dt.tz_localize(None)

        for case_id, seq in seqs.items():
            if case_id not in split_map:
                continue
            last_ts = last_ts_k.get(case_id, pd.NaT)
            case_end = case_end_naive.get(case_id, pd.NaT)
            rem_days = (case_end - last_ts).days if pd.notna(case_end) and pd.notna(last_ts) else np.nan
            rows.append(
                {
                    "case_id": case_id,
                    "prefix_k": k,
                    "sequence": seq,
                    "label": label_map[case_id],
                    "remaining_days": rem_days,
                    "split": split_map[case_id],
                }
            )

    seqs_df = pd.DataFrame(rows)
    seqs_df.to_parquet("data/features/sequences.parquet", index=False)

    # Save activity vocabulary for use in t6_train.py
    with open("data/features/activity_vocab.json", "w") as f:
        json.dump({"act_to_id": act_to_id, "vocab_size": len(act_to_id) + 1}, f)

    print(
        f"     LSTM sequences: {len(seqs_df):,} (case, k) pairs, "
        f"vocab size {len(act_to_id)} activities -> data/features/sequences.parquet"
    )


if __name__ == "__main__":
    main()
