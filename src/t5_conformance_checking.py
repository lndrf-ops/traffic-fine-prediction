"""Task 5: Conformance Checking
- Identification of deviating (sub-)processes
- Definition and verification of compliance rules
- Fitness calculation (Token-Based Replay)
- Saves: outputs/reports/conformance_results.json
"""

import os
import json
import pandas as pd
import pm4py


def main():
    print("=" * 60)
    print("TASK 5: Conformance Checking")
    print("=" * 60)

    df = pd.read_pickle("data/cleaned/df_cleaned.pkl")
    os.makedirs('outputs/reports', exist_ok=True)

    results = {}

    # --- A) FITNESS via Token-Based Replay ---
    print("  1. Computing Fitness (Token-Based Replay)...")
    happy_path_log = pm4py.filter_variants_top_k(df, 10)
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(happy_path_log)

    # Sample for performance
    sample_cases = df['case:concept:name'].drop_duplicates().sample(2000, random_state=42)
    sample_df = df[df['case:concept:name'].isin(sample_cases)]
    fitness = pm4py.fitness_token_based_replay(sample_df, net, initial_marking, final_marking)
    results['fitness'] = fitness
    print(f"     Fitness (% perfectly fitting traces): {fitness['perc_fit_traces']:.2f}%")

    # --- B) COMPLIANCE RULES ---
    print("  2. Checking Compliance Rules...")
    df_sorted = df.sort_values(['case:concept:name', 'time:timestamp']).copy()

    # Rule 1: "Create Fine" must always be the first event
    first_events = df_sorted.groupby('case:concept:name').first()['concept:name']
    rule1_compliance = (first_events == 'Create Fine').mean()
    results['rule1_create_fine_first'] = rule1_compliance
    print(f"     Rule 1 - 'Create Fine' is first event: {rule1_compliance:.2%}")

    # Rule 2: "Payment" and "Send for Credit Collection" must not occur in the same case
    cases_activities = df_sorted.groupby('case:concept:name')['concept:name'].apply(set)
    both_outcomes = cases_activities.apply(lambda x: 'Payment' in x and 'Send for Credit Collection' in x)
    rule2_compliance = 1 - both_outcomes.mean()
    results['rule2_exclusive_outcomes'] = rule2_compliance
    print(f"     Rule 2 - Exclusive outcomes (no Payment + Collection): {rule2_compliance:.2%}")

    # Rule 3: Max Liegezeit zwischen Events < 365 Tage (1 Jahr)
    df_sorted['next_time'] = df_sorted.groupby('case:concept:name')['time:timestamp'].shift(-1)
    df_sorted['wait_days'] = (df_sorted['next_time'] - df_sorted['time:timestamp']).dt.total_seconds() / 86400
    max_waits = df_sorted.groupby('case:concept:name')['wait_days'].max()
    rule3_compliance = (max_waits <= 365).mean()
    results['rule3_max_wait_365d'] = rule3_compliance
    print(f"     Rule 3 - Max wait between steps ≤ 365 days: {rule3_compliance:.2%}")

    # Save results
    with open('outputs/reports/conformance_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✅ Conformance results saved: outputs/reports/conformance_results.json")

    return results


if __name__ == "__main__":
    main()
