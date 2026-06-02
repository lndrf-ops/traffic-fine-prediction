"""Bonus: Generative AI — Synthetic Event Log Generation

Learns a first-order Markov chain from completed RTFM cases and generates
synthetic traces.  Includes a quality evaluation comparing generated traces
against the real event log (activity distribution, trace length distribution,
and directly-follows relation coverage).

Limitation: A first-order Markov model captures only immediate transitions and
cannot reproduce long-range dependencies (e.g., repeated payment cycles).
Higher-order models or neural approaches (e.g., seq2seq) would improve fidelity
but are out of scope for this bonus task.

Saves:
  - outputs/reports/synthetic_event_log.csv
  - outputs/reports/generative_quality_report.json
  - outputs/plots/generative_trace_length_comparison.png
"""

import json
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_transition_matrix(sequences):
    """Build probability dictionary (first-order Markov Model) from real data."""
    transitions = {}
    for seq in sequences:
        for i in range(len(seq)):
            current_state = seq[i]
            next_state = seq[i + 1] if i + 1 < len(seq) else 'END'
            
            # --- FIX: Falls die Zustände fälschlicherweise als Listen reinkommen ---
            if isinstance(current_state, list):
                current_state = current_state[0] if len(current_state) > 0 else 'UNKNOWN'
            if isinstance(next_state, list):
                next_state = next_state[0] if len(next_state) > 0 else 'END'
            # ----------------------------------------------------------------------

            if current_state not in transitions:
                transitions[current_state] = []
            transitions[current_state].append(next_state)
    return transitions


def generate_trace(transitions, start_state='Create Fine', max_length=20):
    """Generate a single synthetic trace using learned transition probabilities."""
    current_state = start_state
    trace = [current_state]

    for _ in range(max_length - 1):
        if current_state not in transitions:
            break
        next_state = random.choice(transitions[current_state])
        if next_state == 'END':
            break
        trace.append(next_state)
        current_state = next_state

    return trace


def evaluate_quality(real_sequences: list[list[str]], synthetic_sequences: list[list[str]]) -> dict:
    """Compare synthetic traces against real traces on key distributional metrics.

    Metrics:
      1. Activity frequency distribution (Jensen-Shannon divergence)
      2. Trace length distribution (mean, std, KS-like comparison)
      3. Directly-follows coverage (% of real DF pairs reproduced)
    """
    # --- Activity frequency ---
    real_acts = Counter(a for seq in real_sequences for a in seq)
    synth_acts = Counter(a for seq in synthetic_sequences for a in seq)
    all_acts = sorted(set(real_acts.keys()) | set(synth_acts.keys()))

    real_total = sum(real_acts.values())
    synth_total = sum(synth_acts.values())
    real_dist = np.array([real_acts.get(a, 0) / real_total for a in all_acts])
    synth_dist = np.array([synth_acts.get(a, 0) / synth_total for a in all_acts])

    # Jensen-Shannon divergence (symmetric KL)
    m = 0.5 * (real_dist + synth_dist)
    # Avoid log(0)
    eps = 1e-12
    kl_rm = np.sum(real_dist * np.log((real_dist + eps) / (m + eps)))
    kl_sm = np.sum(synth_dist * np.log((synth_dist + eps) / (m + eps)))
    jsd = 0.5 * (kl_rm + kl_sm)

    # --- Trace lengths ---
    real_lengths = [len(s) for s in real_sequences]
    synth_lengths = [len(s) for s in synthetic_sequences]

    # --- Directly-follows coverage ---
    def get_df_pairs(sequences):
        pairs = set()
        for seq in sequences:
            for i in range(len(seq) - 1):
                pairs.add((seq[i], seq[i + 1]))
        return pairs

    real_df = get_df_pairs(real_sequences)
    synth_df = get_df_pairs(synthetic_sequences)
    df_coverage = len(real_df & synth_df) / len(real_df) if real_df else 0.0

    return {
        "activity_jsd": round(float(jsd), 6),
        "activity_jsd_interpretation": "0=identical, 0.69=max divergence (ln2)",
        "real_trace_length_mean": round(np.mean(real_lengths), 2),
        "real_trace_length_std": round(np.std(real_lengths), 2),
        "synth_trace_length_mean": round(np.mean(synth_lengths), 2),
        "synth_trace_length_std": round(np.std(synth_lengths), 2),
        "directly_follows_coverage": round(df_coverage, 4),
        "real_df_pairs": len(real_df),
        "synth_df_pairs": len(synth_df),
        "activity_distribution": {
            a: {"real_pct": round(real_dist[i] * 100, 2), "synth_pct": round(synth_dist[i] * 100, 2)}
            for i, a in enumerate(all_acts)
        },
    }


def plot_trace_length_comparison(real_sequences, synthetic_sequences, save_path):
    """Histogram comparing trace length distributions."""
    real_lengths = [len(s) for s in real_sequences]
    synth_lengths = [len(s) for s in synthetic_sequences]

    fig, ax = plt.subplots(figsize=(8, 4))
    max_len = max(max(real_lengths), max(synth_lengths))
    bins = range(1, min(max_len + 2, 25))

    ax.hist(real_lengths, bins=bins, alpha=0.6, label="Real", density=True, color="#636EFA")
    ax.hist(synth_lengths, bins=bins, alpha=0.6, label="Synthetic", density=True, color="#EF553B")
    ax.set_xlabel("Trace Length (number of events)")
    ax.set_ylabel("Density")
    ax.set_title("Trace Length Distribution: Real vs. Synthetic (Markov)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("BONUS: Generative AI (Synthetic Event Log)")
    print("=" * 60)

    os.makedirs('outputs/reports', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    random.seed(42)

    # 1. Load real sequences (grouped by case) — support two formats:
    #    - case-level `completed_cases` with a `trace` column (list of activities)
    #    - event-level DataFrame with one row per event and a `concept:name` column
    completed_cases = pd.read_pickle("data/cleaned/completed_cases.pkl")

    def _flatten_group_values(values):
        flat = []
        for v in values:
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(v)
        return flat

    if 'trace' in completed_cases.columns:
        # Case-level format produced by t3_data_cleaning.py
        col = completed_cases['trace']
        print("[diagnostic] completed_cases contains 'trace' column (case-level).")
        print("[diagnostic] trace dtype:", col.dtype)
        print("[diagnostic] sample traces (first 10):")
        for i, v in enumerate(col.head(10).tolist(), 1):
            print(f"  {i}: {repr(v)}")
        # Ensure lists
        real_sequences = col.apply(lambda x: x if isinstance(x, list) else [x]).tolist()
    elif 'concept:name' in completed_cases.columns:
        # Event-level format: perform diagnostics on `concept:name`
        col = completed_cases["concept:name"]
        print("[diagnostic] completed_cases contains event-level 'concept:name' column.")
        print("[diagnostic] concept:name dtype:", col.dtype)
        type_counts = col.apply(lambda x: type(x)).value_counts()
        print("[diagnostic] types in concept:name:\n", type_counts.to_dict())
        print("[diagnostic] sample values (first 10):")
        for i, v in enumerate(col.head(10).tolist(), 1):
            print(f"  {i}: {repr(v)}")
        if col.apply(lambda x: isinstance(x, list)).any():
            list_lens = col[col.apply(lambda x: isinstance(x, list))].apply(len)
            print("[diagnostic] list-entry length stats (if any):\n", list_lens.describe().to_dict())

        # Handle possible list entries inside `concept:name` or nested structures
        if col.apply(lambda x: isinstance(x, list)).any():
            case_counts = completed_cases.groupby("case:concept:name").size()
            if (case_counts == 1).all():
                real_sequences = completed_cases["concept:name"].apply(lambda x: x if isinstance(x, list) else [x]).tolist()
            else:
                real_sequences = completed_cases.groupby("case:concept:name")["concept:name"].apply(_flatten_group_values).tolist()
        else:
            real_sequences = completed_cases.groupby("case:concept:name")["concept:name"].apply(list).tolist()
    else:
        raise RuntimeError("Loaded completed_cases does not contain 'trace' nor 'concept:name' columns — cannot build sequences")

    # 2. Train first-order Markov Model
    print("  Training first-order Markov Chain model...")
    transition_matrix = build_transition_matrix(real_sequences)
    print(f"    States learned: {len(transition_matrix)}")

    # 3. Generate synthetic cases (same count as real for fair comparison)
    num_synthetic = 1000
    print(f"  Generating {num_synthetic} synthetic cases...")

    synthetic_sequences = []
    synthetic_log = []
    for i in range(num_synthetic):
        trace = generate_trace(transition_matrix, start_state='Create Fine')
        synthetic_sequences.append(trace)
        for step, activity in enumerate(trace):
            synthetic_log.append({
                'case:concept:name': f"SYNTH_{i + 1}",
                'concept:name': activity,
                'event_position': step + 1
            })

    df_synthetic = pd.DataFrame(synthetic_log)
    save_path = "outputs/reports/synthetic_event_log.csv"
    df_synthetic.to_csv(save_path, index=False)
    print(f"  Synthetic log generated: {len(df_synthetic):,} events from {num_synthetic} cases")

    # 4. Quality evaluation
    print("\n  Evaluating synthetic trace quality...")
    quality = evaluate_quality(real_sequences, synthetic_sequences)

    report_path = "outputs/reports/generative_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(quality, f, indent=2)

    print(f"    Activity JSD:              {quality['activity_jsd']:.4f} (lower=better)")
    print(f"    DF-relation coverage:      {quality['directly_follows_coverage']:.1%}")
    print(f"    Real trace length:         {quality['real_trace_length_mean']:.1f} ± {quality['real_trace_length_std']:.1f}")
    print(f"    Synthetic trace length:    {quality['synth_trace_length_mean']:.1f} ± {quality['synth_trace_length_std']:.1f}")

    # 5. Comparison plot
    plot_path = "outputs/plots/generative_trace_length_comparison.png"
    plot_trace_length_comparison(real_sequences, synthetic_sequences, plot_path)

    print(f"\n  ✅ Saved: {save_path}")
    print(f"     Saved: {report_path}")
    print(f"     Saved: {plot_path}")


if __name__ == "__main__":
    main()
