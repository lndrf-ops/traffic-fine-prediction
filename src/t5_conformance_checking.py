"""Task 5: Conformance Checking

Implements the 5 conformance rules from docs/conformance_rules.md:

Structural (Token-Based Replay on top-10-variant Petri net):
  Rule 1: Payment must not occur before Create Fine
  Rule 2: Send Appeal to Prefecture requires prior Insert Fine Notification
  Rule 3: Cases with Send for Credit Collection must not contain subsequent Payment

Temporal / Data-Aware (custom iteration):
  Rule 4: Add penalty >= 60 days after Send Fine  [Italian Codice della Strada Art. 203]
  Rule 5: Once cumulative Payment >= totalPaymentAmount, no further Payment events

Method reference: van der Aalst (2016), Process Mining: Data Science in Action, Ch. 8.
Saves: outputs/reports/conformance_results.json
"""

import json
import os

import pandas as pd
import pm4py


def check_rule1(df_sorted: pd.DataFrame) -> dict:
    """Payment must not occur before Create Fine."""
    violations = 0
    for case_id, group in df_sorted.groupby("case:concept:name"):
        acts = group["concept:name"].tolist()
        if "Payment" in acts and "Create Fine" in acts:
            if acts.index("Payment") < acts.index("Create Fine"):
                violations += 1
    total = df_sorted["case:concept:name"].nunique()
    return {
        "rule": "Payment not before Create Fine",
        "source": "de Leoni & Mannhardt (2015)",
        "total_cases": total,
        "violations": violations,
        "compliance_rate": round(1 - violations / total, 6),
    }


def check_rule2(df_sorted: pd.DataFrame) -> dict:
    """Send Appeal to Prefecture requires prior Insert Fine Notification."""
    violations = 0
    for case_id, group in df_sorted.groupby("case:concept:name"):
        acts = group["concept:name"].tolist()
        if "Send Appeal to Prefecture" in acts:
            appeal_pos = acts.index("Send Appeal to Prefecture")
            has_prior_notification = any(
                a == "Insert Fine Notification" for a in acts[:appeal_pos]
            )
            if not has_prior_notification:
                violations += 1
    appeal_cases = df_sorted[
        df_sorted["concept:name"] == "Send Appeal to Prefecture"
    ]["case:concept:name"].nunique()
    return {
        "rule": "Send Appeal to Prefecture requires prior Insert Fine Notification",
        "source": "Mannhardt et al. (2016), Table 3, Constraint C3",
        "cases_with_appeal": appeal_cases,
        "violations": violations,
        "compliance_rate": round(1 - violations / appeal_cases, 6) if appeal_cases else 1.0,
    }


def check_rule3(df_sorted: pd.DataFrame) -> dict:
    """Cases with Send for Credit Collection must not contain subsequent Payment."""
    violations = 0
    collection_cases = 0
    for case_id, group in df_sorted.groupby("case:concept:name"):
        acts = group["concept:name"].tolist()
        if "Send for Credit Collection" in acts:
            collection_cases += 1
            collection_pos = acts.index("Send for Credit Collection")
            has_subsequent_payment = any(
                a == "Payment" for a in acts[collection_pos + 1:]
            )
            if has_subsequent_payment:
                violations += 1
    return {
        "rule": "No Payment after Send for Credit Collection",
        "source": "Mannhardt et al. (2016), implicit; process description",
        "cases_with_credit_collection": collection_cases,
        "violations": violations,
        "compliance_rate": round(1 - violations / collection_cases, 6) if collection_cases else 1.0,
    }


def check_rule4(df_sorted: pd.DataFrame) -> dict:
    """Add penalty >= 60 days after Send Fine (Codice della Strada Art. 203)."""
    violations = 0
    applicable = 0
    for case_id, group in df_sorted.groupby("case:concept:name"):
        acts = group["concept:name"].tolist()
        if "Add penalty" in acts and "Send Fine" in acts:
            applicable += 1
            send_fine_ts = group[group["concept:name"] == "Send Fine"]["time:timestamp"].min()
            add_penalty_ts = group[group["concept:name"] == "Add penalty"]["time:timestamp"].min()
            delta_days = (add_penalty_ts - send_fine_ts).days
            if delta_days < 60:
                violations += 1
    return {
        "rule": "Add penalty >= 60 days after Send Fine",
        "source": "Mannhardt et al. (2016), §6.2; Italian Codice della Strada Art. 203",
        "applicable_cases": applicable,
        "violations": violations,
        "compliance_rate": round(1 - violations / applicable, 6) if applicable else 1.0,
    }


def check_rule5(df_sorted: pd.DataFrame) -> dict:
    """Once cumulative Payment >= totalPaymentAmount, no further Payment events."""
    violations = 0
    applicable = 0
    for case_id, group in df_sorted.groupby("case:concept:name"):
        payment_events = group[group["concept:name"] == "Payment"]
        if payment_events.empty:
            continue
        total_due = group["totalPaymentAmount"].dropna()
        if total_due.empty:
            continue
        applicable += 1
        total_due_val = total_due.iloc[0]
        cumulative = 0.0
        overpaid = False
        for _, row in payment_events.iterrows():
            cumulative += row.get("amount", 0) or 0
            if cumulative >= total_due_val and not overpaid:
                overpaid = True
                continue
            if overpaid:
                violations += 1
                break
    return {
        "rule": "No Payment after cumulative amount >= totalPaymentAmount",
        "source": "de Leoni, van der Aalst & Dees (2016), §7",
        "applicable_cases": applicable,
        "violations": violations,
        "compliance_rate": round(1 - violations / applicable, 6) if applicable else 1.0,
    }


def token_based_fitness(df: pd.DataFrame) -> dict:
    """Token-Based Replay fitness on the top-10-variant Petri net."""
    happy_path_log = pm4py.filter_variants_top_k(df, 10)
    net, im, fm = pm4py.discover_petri_net_inductive(happy_path_log)

    sample = df[
        df["case:concept:name"].isin(
            df["case:concept:name"].drop_duplicates().sample(2000, random_state=42)
        )
    ]
    fitness = pm4py.fitness_token_based_replay(sample, net, im, fm)
    return {
        "method": "Token-Based Replay (top-10-variant Petri net, 2000-case sample)",
        "source": "van der Aalst (2016), Ch. 8",
        "perc_fit_traces": round(fitness["perc_fit_traces"], 4),
        "average_trace_fitness": round(fitness["average_trace_fitness"], 4),
    }


def main():
    print("=" * 60)
    print("TASK 5: Conformance Checking")
    print("=" * 60)

    df = pd.read_pickle("data/cleaned/df_cleaned.pkl")
    df_sorted = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    os.makedirs("outputs/reports", exist_ok=True)

    results = {}

    print("  1. Token-Based Replay fitness...")
    results["token_based_fitness"] = token_based_fitness(df)
    print(f"     Fit traces: {results['token_based_fitness']['perc_fit_traces']:.2f}%")

    print("  2. Rule 1 — Payment not before Create Fine...")
    results["rule1"] = check_rule1(df_sorted)
    print(f"     Compliance: {results['rule1']['compliance_rate']:.2%}  "
          f"({results['rule1']['violations']} violations)")

    print("  3. Rule 2 — Appeal requires prior notification...")
    results["rule2"] = check_rule2(df_sorted)
    print(f"     Compliance: {results['rule2']['compliance_rate']:.2%}  "
          f"({results['rule2']['violations']} violations)")

    print("  4. Rule 3 — No Payment after Credit Collection...")
    results["rule3"] = check_rule3(df_sorted)
    print(f"     Compliance: {results['rule3']['compliance_rate']:.2%}  "
          f"({results['rule3']['violations']} violations)")

    print("  5. Rule 4 — Add penalty >= 60 days after Send Fine...")
    results["rule4"] = check_rule4(df_sorted)
    print(f"     Compliance: {results['rule4']['compliance_rate']:.2%}  "
          f"({results['rule4']['violations']} / {results['rule4']['applicable_cases']} applicable cases)")

    print("  6. Rule 5 — No Payment after full settlement...")
    results["rule5"] = check_rule5(df_sorted)
    print(f"     Compliance: {results['rule5']['compliance_rate']:.2%}  "
          f"({results['rule5']['violations']} violations)")

    with open("outputs/reports/conformance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Conformance results saved: outputs/reports/conformance_results.json")


if __name__ == "__main__":
    main()
