"""Task 5: Conformance Checking

Implements 5 domain-specific conformance rules derived from the Italian
Codice della Strada (CdS) — the statutory framework governing road traffic fines.

Rules:
  1. 90-Day Statutory Deadline (Art. 201 CdS): Create Fine -> Send Fine <= 90 days
  2. 60-Day Penalty Grace Period (Art. 202§1, 203§3 CdS): Add penalty >= 60 days after notification
  3. Due Process / Appeal Prerequisite (Art. 203§1 CdS): Appeal requires prior notification
  4. Duty to Inform (Art. 204§2 CdS): Prefecture result must be followed by notification to offender
  5. Statutory Payment Accuracy (Art. 202§1, 203§3 CdS): No payment after debt is settled

Additionally runs Token-Based Replay fitness on a discovered Petri net.

Saves: outputs/reports/conformance_results.json
"""

import json
import os

import pandas as pd
import pm4py


def check_rule1_90day_deadline(df_sorted: pd.DataFrame) -> dict:
    """Rule 1: 90-Day Statutory Deadline (Art. 201 CdS).

    The time between Create Fine and Send Fine must not exceed 90 days.
    Failure makes the fine legally contestable and may result in annulment.
    """
    violations = 0
    applicable = 0
    violation_days = []
    for case_id, group in df_sorted.groupby("case:concept:name"):
        acts = group["concept:name"].tolist()
        if "Create Fine" in acts and "Send Fine" in acts:
            applicable += 1
            create_ts = group[group["concept:name"] == "Create Fine"]["time:timestamp"].iloc[0]
            send_ts = group[group["concept:name"] == "Send Fine"]["time:timestamp"].min()
            delta_days = (send_ts - create_ts).days
            if delta_days > 90:
                violations += 1
                violation_days.append(delta_days)
    return {
        "rule": "90-Day Statutory Deadline: Create Fine -> Send Fine <= 90 days",
        "source": "Art. 201 CdS -- notification obligation within 90 days of ascertainment",
        "applicable_cases": applicable,
        "violations": violations,
        "compliance_rate": round(1 - violations / applicable, 6) if applicable else 1.0,
        "avg_violation_days": round(sum(violation_days) / len(violation_days), 1) if violation_days else None,
    }


def check_rule2_60day_penalty(df_sorted: pd.DataFrame) -> dict:
    """Rule 2: 60-Day Penalty Grace Period (Art. 202§1, 203§3 CdS).

    Add penalty must not occur earlier than 60 days after the notification
    (Insert Fine Notification or Send Fine), respecting the citizen's payment window.
    """
    violations = 0
    applicable = 0
    for case_id, group in df_sorted.groupby("case:concept:name"):
        acts = group["concept:name"].tolist()
        if "Add penalty" not in acts:
            continue
        # Find the notification timestamp (Insert Fine Notification preferred, else Send Fine)
        notif_events = group[group["concept:name"].isin(["Insert Fine Notification", "Send Fine"])]
        if notif_events.empty:
            continue
        applicable += 1
        notif_ts = notif_events["time:timestamp"].min()
        penalty_ts = group[group["concept:name"] == "Add penalty"]["time:timestamp"].min()
        delta_days = (penalty_ts - notif_ts).days
        if delta_days < 60:
            violations += 1
    return {
        "rule": "60-Day Penalty Grace Period: Add penalty >= 60 days after notification",
        "source": "Art. 202par1, 203par3 CdS -- 60-day payment window before penalty escalation",
        "applicable_cases": applicable,
        "violations": violations,
        "compliance_rate": round(1 - violations / applicable, 6) if applicable else 1.0,
    }


def check_rule3_appeal_prerequisite(df_sorted: pd.DataFrame) -> dict:
    """Rule 3: Due Process / Appeal Prerequisite (Art. 203§1 CdS).

    Send Appeal to Prefecture must be preceded by Insert Fine Notification
    (or Send Fine as minimum notification). An appeal cannot legally exist
    without a prior formal notification.
    """
    violations = 0
    applicable = 0
    for case_id, group in df_sorted.groupby("case:concept:name"):
        acts = group["concept:name"].tolist()
        if "Send Appeal to Prefecture" not in acts:
            continue
        applicable += 1
        appeal_pos = acts.index("Send Appeal to Prefecture")
        has_prior_notification = any(
            a in ("Insert Fine Notification", "Send Fine") for a in acts[:appeal_pos]
        )
        if not has_prior_notification:
            violations += 1
    return {
        "rule": "Due Process: Appeal requires prior notification",
        "source": "Art. 203par1 CdS -- appeal within 60 days of notification/communication",
        "applicable_cases": applicable,
        "violations": violations,
        "compliance_rate": round(1 - violations / applicable, 6) if applicable else 1.0,
    }


def check_rule4_duty_to_inform(df_sorted: pd.DataFrame) -> dict:
    """Rule 4: Duty to Inform (Art. 204§2 CdS).

    In completed cases, Receive Result Appeal from Prefecture must be followed
    by Notify Result Appeal to Offender. Right-censored (open) cases are excluded.
    """
    violations = 0
    applicable = 0
    terminal_activities = {"Payment", "Send for Credit Collection"}
    for case_id, group in df_sorted.groupby("case:concept:name"):
        acts = group["concept:name"].tolist()
        # Only consider completed cases
        if not any(a in terminal_activities for a in acts):
            continue
        if "Receive Result Appeal from Prefecture" not in acts:
            continue
        applicable += 1
        if "Notify Result Appeal to Offender" not in acts:
            violations += 1
        else:
            # Ensure notification comes after receiving the result
            receive_pos = acts.index("Receive Result Appeal from Prefecture")
            notify_pos = acts.index("Notify Result Appeal to Offender")
            if notify_pos < receive_pos:
                violations += 1
    return {
        "rule": "Duty to Inform: Prefecture result must be notified to offender",
        "source": "Art. 204par2 CdS -- obligation to notify the Prefect's decision",
        "applicable_cases": applicable,
        "violations": violations,
        "compliance_rate": round(1 - violations / applicable, 6) if applicable else 1.0,
    }


def check_rule5_payment_accuracy(df_sorted: pd.DataFrame) -> dict:
    """Rule 5: Statutory Payment Accuracy (Art. 202§1, 203§3 CdS).

    Once cumulative payments meet or exceed the required amount (totalPaymentAmount),
    no further Payment events should occur.
    """
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
        settled = False
        for _, row in payment_events.iterrows():
            cumulative += row.get("paymentAmount", 0) or row.get("amount", 0) or 0
            if cumulative >= total_due_val and not settled:
                settled = True
                continue
            if settled:
                violations += 1
                break
    return {
        "rule": "Payment Accuracy: No payment after statutory debt is settled",
        "source": "Art. 202par1, 203par3 CdS -- statutory fine amounts and discount/penalty rules",
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

    print("  2. Rule 1 -- 90-Day Statutory Deadline (Art. 201 CdS)...")
    results["rule1"] = check_rule1_90day_deadline(df_sorted)
    print(f"     Compliance: {results['rule1']['compliance_rate']:.2%}  "
          f"({results['rule1']['violations']} / {results['rule1']['applicable_cases']} applicable)")

    print("  3. Rule 2 -- 60-Day Penalty Grace Period (Art. 202par1, 203par3 CdS)...")
    results["rule2"] = check_rule2_60day_penalty(df_sorted)
    print(f"     Compliance: {results['rule2']['compliance_rate']:.2%}  "
          f"({results['rule2']['violations']} / {results['rule2']['applicable_cases']} applicable)")

    print("  4. Rule 3 -- Due Process / Appeal Prerequisite (Art. 203par1 CdS)...")
    results["rule3"] = check_rule3_appeal_prerequisite(df_sorted)
    print(f"     Compliance: {results['rule3']['compliance_rate']:.2%}  "
          f"({results['rule3']['violations']} / {results['rule3']['applicable_cases']} applicable)")

    print("  5. Rule 4 -- Duty to Inform (Art. 204par2 CdS)...")
    results["rule4"] = check_rule4_duty_to_inform(df_sorted)
    print(f"     Compliance: {results['rule4']['compliance_rate']:.2%}  "
          f"({results['rule4']['violations']} / {results['rule4']['applicable_cases']} applicable)")

    print("  6. Rule 5 -- Statutory Payment Accuracy (Art. 202par1, 203par3 CdS)...")
    results["rule5"] = check_rule5_payment_accuracy(df_sorted)
    print(f"     Compliance: {results['rule5']['compliance_rate']:.2%}  "
          f"({results['rule5']['violations']} / {results['rule5']['applicable_cases']} applicable)")

    with open("outputs/reports/conformance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Conformance results saved: outputs/reports/conformance_results.json")


if __name__ == "__main__":
    main()
