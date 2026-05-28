"""Bonus: Generative AI
- Generation of synthetic event logs / traces
- Stochastic Markov chain model
- Saves: outputs/reports/synthetic_event_log.csv
"""

import os
import random
import pandas as pd


def build_transition_matrix(sequences):
    """Build probability dictionary (Markov Model) from real data"""
    transitions = {}
    for seq in sequences:
        for i in range(len(seq)):
            current_state = seq[i]
            next_state = seq[i + 1] if i + 1 < len(seq) else 'END'
            if current_state not in transitions:
                transitions[current_state] = []
            transitions[current_state].append(next_state)
    return transitions


def generate_trace(transitions, start_state='Create Fine', max_length=20):
    """Generate synthetic trace based on learned transition probabilities"""
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


def main():
    print("=" * 60)
    print("BONUS: Generative AI (Synthetic Event Log)")
    print("=" * 60)

    os.makedirs('outputs/reports', exist_ok=True)

    # 1. Load real sequences
    completed_cases = pd.read_pickle("data/cleaned/completed_cases.pkl")
    real_sequences = completed_cases['concept:name'].tolist()

    # 2. Train Markov Model
    print("  Training stochastic Markov Chain model...")
    transition_matrix = build_transition_matrix(real_sequences)

    # 3. Generate synthetic cases
    num_synthetic = 1000
    print(f"  Generating {num_synthetic} synthetic cases...")

    synthetic_log = []
    for i in range(num_synthetic):
        trace = generate_trace(transition_matrix, start_state='Create Fine')
        for step, activity in enumerate(trace):
            synthetic_log.append({
                'case:concept:name': f"SYNTH_{i + 1}",
                'concept:name': activity,
                'event_position': step + 1
            })

    df_synthetic = pd.DataFrame(synthetic_log)
    save_path = "outputs/reports/synthetic_event_log.csv"
    df_synthetic.to_csv(save_path, index=False)
    print(f"  ✅ Synthetic log generated: {len(df_synthetic):,} events from {num_synthetic} cases")
    print(f"     Saved: {save_path}")


if __name__ == "__main__":
    main()
