# Prefix-Length Strategy

Predictive Process Monitoring predicts outcomes for *ongoing* cases — from
the first *k* events ("prefix") only. Follows Teinemaa et al. (2019), §3.2
"Trace Bucketing".

## Evaluated Prefix Lengths

`k ∈ {2, 3, 5, 8}`

| k | Rationale |
|---|---|
| 2 | Earliest sensible point (after `Create Fine` + `Send Fine`) |
| 3 | Includes first reaction (Payment / Notification) |
| 5 | Typical mid-process state (RTFM median ~5 events) |
| 8 | Late state (RTFM 90th percentile ~8–9 events) |

Cases with fewer than k events are **excluded from that bucket**. Per-bucket
sample sizes are reported in the evaluation.

## Bucketing Strategy

### Classical Models (LogReg, RF, XGBoost) → Prefix-Length Bucketing
**One model trained per k.**
Rationale: each k has a different feature distribution; per-bucket models
achieve best benchmark scores (Teinemaa 2019).

### LSTM → Single Model
**One model for all prefix lengths.**
Rationale: LSTMs natively handle variable-length sequences via padding and
masking; bucketing would discard their core architectural strength.

## Implementation

- Prefix extraction in `t6_feature_engineering.py::build_prefixes(df, k)`
- Output: `data/features/prefix_k{2,3,5,8}.parquet` (classical) and
  `data/features/sequences.parquet` (LSTM)