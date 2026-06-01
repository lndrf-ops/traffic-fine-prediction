# Leakage Guardrails

Predictive Process Monitoring is highly susceptible to target leakage.
The following checks are enforced in `t6_feature_engineering.py` and
verified by assertions before training.

## 1. Outcome-Revealing Activities

`Payment` and `Send for Credit Collection` define the label — must NEVER
appear in the prefix.

**Enforcement**: when building prefix of length k, truncate at
min(k, position of first outcome activity − 1). If an outcome activity
occurs within the first k events, exclude the case from that bucket.

## 2. Cumulative Payload Attributes

`expense`, `paymentAmount`, `totalPaymentAmount` are cumulative across
events. Using their final case-level value at prefix k leaks the future.

**Enforcement**: aggregate only over events e₁ ... eₖ, never over the full case.

## 3. Case-Level Aggregates

Statistics like `case_duration`, `num_events_total`, `has_appeal` (computed
over the full case) leak the future.

**Enforcement**: every case-level feature is recomputed at prefix k as a
prefix-aware version (e.g., `has_appeal_so_far`, `events_so_far`, `duration_so_far`).

## 4. Temporal Overlap Between Train and Test

Cases bridging the split boundary can leak via shared context.

**Enforcement**: strict temporal split — cases whose `[start, end]` crosses
the boundary are dropped ("debiased split", Teinemaa 2019).

## 5. Categorical Encoding

Target encoding of high-cardinality categoricals (e.g., `article`) must fit
on training data only.

**Enforcement**: encoders pickled after fitting; test pipeline loads them read-only.

## 6. Normalization / Scaling

Scalers fit on full data leak the test distribution.

**Enforcement**: all scalers fit on train fold only; transform applied to val/test.

## References

- Teinemaa et al. (2019), §4 "Common Pitfalls"
- Weytjens & De Weerdt (2022), "Creating Unbiased Public Benchmarks for PPM"