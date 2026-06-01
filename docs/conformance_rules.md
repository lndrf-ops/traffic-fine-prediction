# Conformance Checking Rules

Top conformance constraints for RTFM, selected from the established literature.
Implemented in `t5_conformance_checking.py`.

## Structural Constraints (Token-Based Replay)

### 1. `Payment` must not occur before `Create Fine`
Trivial causality; sanity check that should hold 100%. Violations indicate
data quality issues.
*Source: de Leoni & Mannhardt (2015), dataset description.*

### 2. `Send Appeal to Prefecture` requires prior `Insert Fine Notification`
One cannot appeal a fine without official notification.
*Source: Mannhardt et al. (2016), Table 3, Constraint C3.*

### 3. Cases with `Send for Credit Collection` must not contain subsequent `Payment`
Credit collection is the terminal failure state.
*Source: implicit in Mannhardt et al. (2016) and the official process description.*

## Temporal / Data-Aware Constraints (custom implementation)

### 4. `Add penalty` must occur ≥60 days after `Send Fine` ⭐
Italian traffic law (Codice della Strada Art. 203). Goldstandard RTFM rule.
*Source: Mannhardt et al. (2016), §6.2.*

### 5. Once cumulative `Payment` ≥ `totalPaymentAmount`, no further `Payment` events
Full payment closes the case.
*Source: de Leoni, van der Aalst & Dees (2016), §7.*

## Methodology

| Rule | Method | pm4py Function |
|---|---|---|
| #1–#3 | Token-Based Replay | `pm4py.conformance_diagnostics_token_based_replay` |
| #1–#3 (deep dive) | Alignment-Based | `pm4py.conformance_diagnostics_alignments` |
| #4–#5 | Custom iteration | — (pm4py has no quantitative time-constraint support) |

Token-based replay is fast; alignment-based is more precise but slower —
use alignments only for the top deviations identified by replay.

*Method reference: van der Aalst (2016), Process Mining: Data Science in
Action (2nd ed.), Chapter 8.*s