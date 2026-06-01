# Project Review — Road Traffic Fine Management Process

## Review Checklist
- [x] Phase 0: Setup & Project Structure
- [x] Phase 1: Data Loading
- [x] Phase 2: Descriptive Analysis
- [x] Phase 3: Data Cleaning
- [x] Phase 4: Process Discovery
- [x] Phase 5: Conformance Checking
- [x] Phase 6: Feature Engineering & Training
- [x] Phase 7: Evaluation & Interpretability
- [x] Phase 8: Streamlit App & Bonus Extensions
- [x] Gesamtaufbau & Stringenz reviewed

---

## Phase 0: Setup & Project Structure

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 6 |
| 2. Code Quality & Appropriateness | 7 |
| 3. Storage of Relevant Results | 5 |

**Feedback:** The project structure is clean and logical (`src/`, `data/`, `outputs/`, `app/`). `CLAUDE.md` is well-written and serves as effective project documentation. `run_pipeline.py` with `--from` and `--only` flags is a nice touch. However, several planned items are **missing**: no `configs/` directory (all hyperparameters hardcoded in scripts rather than externalized as per plan.md), no `tests/` directory (plan required "≥1 sanity check per phase"), no `docs/decisions.md` (plan required documenting decisions per phase). The `docs/` folder has useful specs (leakage_guardrails, prefix_strategy, etc.) which partially compensate. `requirements.txt` exists but there is no `pyproject.toml` or lockfile for reproducibility.

---

## Phase 1: Data Loading

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 7 |
| 3. Storage of Relevant Results | 7 |

**Feedback:** Simple, effective, does what it needs to do. Loads XES via pm4py, prints basic stats, saves to pickle. The fallback between `.xes.gz` and `.xes` is a practical touch. Minor issues: (1) saves raw data to `data/cleaned/` which is semantically misleading — raw data should go to `data/raw/` or `data/interim/`; (2) no schema validation or assertion on expected columns; (3) task numbering in print header is inconsistent (script is t1 but print says "TASK 1" while t2 says "TASK 3").

---

## Phase 2: Descriptive Analysis

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 7 |
| 2. Code Quality & Appropriateness | 7 |
| 3. Storage of Relevant Results | 8 |

**Feedback:** Solid column profiling with fill rates, min/max/median, unique counts — correctly run on raw pre-cleaning data. Low-quality column identification with threshold is good practice. JSON report saved with structured data. Dotted chart generated. What's missing: no activity frequency analysis, no case duration distribution at this stage, no explicit documentation of the "timestamps are date-only" limitation in the output (only mentioned in plan.md). The print header says "TASK 3" but it's actually Phase 2 in the pipeline — confusing numbering throughout.

---

## Phase 3: Data Cleaning

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 7 |
| 3. Storage of Relevant Results | 7 |

**Feedback:** Good: duplicate removal, column drops justified by Phase 2 profiling, labelling logic with clear three-rule priority (well-documented in code comments). The `determine_outcome()` function correctly handles the "collection wins over payment" semantics. Case-level attribute extraction is useful. Issues: (1) `event_position` is computed but unclear if used downstream; (2) no assertion or logging of how many cases are excluded (only printed); (3) `completed_cases.pkl` stores activity lists as Python lists inside a DataFrame column — fragile serialization choice.

---

## Phase 4: Process Discovery

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 6 |
| 3. Storage of Relevant Results | 8 |

**Feedback:** Comprehensive: bottleneck analysis (mean + median), case durations, top-5 variants, performance spectrum, and Petri net discovery. All plots saved. The performance spectrum implementation (manual matplotlib scatter for 150 sampled cases) is creative. However: (1) hardcoded magic numbers (`< 730` days filter, `sample(150)`), no comments explaining choices; (2) all plot styling is inline and repetitive — could use a shared plot utility; (3) no organizational perspective despite plan.md listing it (though this is justified since `org:resource` was dropped); (4) Petri net is saved as `.dot` and `.png` but no quality metrics (fitness, precision) reported here — that's deferred to Phase 5 which is appropriate.

---

## Phase 5: Conformance Checking

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 8 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** Strongest module in the project. Five well-defined conformance rules with literature sources cited in code (Mannhardt et al., 2016; Codice della Strada Art. 203). Clean function-per-rule structure. Both structural (ordering constraints) and temporal/data-aware (60-day rule, payment completeness) rules implemented. Results saved as JSON with compliance rates. The rules are domain-grounded and non-trivial. Minor criticism: looping over `groupby` with Python for-loops is O(n_cases) with high constant — vectorized approaches would be more efficient for 150k cases, though runtime is acceptable for a university project.

---

## Phase 6: Feature Engineering & Training

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 7 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** 

**Feature Engineering (t6_feature_engineering.py):** Excellent implementation of the Teinemaa (2019) prefix-based approach. Temporal split with debiasing (boundary-crossing cases removed) is correctly implemented and well-documented. The CF/DA split is clean after recent refactoring (CF = binary activity indicators only; DA adds temporal + payload). All prefix parquet files, split indices, and LSTM sequences saved. Leakage guardrail (excluding outcome-revealing activities from prefixes) is correctly enforced.

**Training (t6_train.py):** All planned models implemented: baselines, LogReg, RF, XGBoost, LSTM. Seeds fixed. XGBoost uses early stopping with validation set — correct. `class_weight="balanced"` and `scale_pos_weight` applied per plan. LSTM architecture is reasonable (embedding → LSTM → dropout → linear). Feature column lists persisted alongside models (critical for inference).

Issues: (1) `n_jobs=1` with comment "FIX" — works around macOS segfault but should note this is platform-specific; (2) no hyperparameter tuning (grid search / Optuna) — all hyperparameters manually chosen. For an MSc project, at least documenting why defaults were kept would strengthen it; (3) the LSTM uses a simple activity-embedding approach without the temporal/DA features — it only gets the CF variant, limiting fair comparison; (4) `OBJC_DISABLE_INITIALIZE_FORK_SAFETY` is a hack documented only in comments.

---

## Phase 7: Evaluation & Interpretability

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 7 |
| 3. Storage of Relevant Results | 9 |

**Feedback:**

**Evaluation (t6_evaluate.py):** Comprehensive metric set: AUC-ROC, accuracy, F1/precision/recall per class for outcome; MAE/RMSE for remaining time. Overfitting assessment (train vs test gap) saved separately. Evaluation plots generated. LSTM evaluation correctly loads sequences and uses same test cases. All results persisted to `evaluation_results.json`.

**Interpretability (t6_interpretability.py):** SHAP TreeExplainer on RF for all 8 combinations (2 variants × 4 prefix lengths). Summary dot plots saved at 300 DPI. Clean, focused module.

Issues: (1) Overfitting assessment shows `gap: -0.053` for LogReg/RF/XGB at k=2 — **test AUC higher than train AUC** which is suspicious and suggests a bug in train-set metric computation (possibly evaluating on a subset or the train AUC was computed before fitting); (2) SHAP uses RF models but the app now loads XGBoost — inconsistency between interpretability module and live prediction; (3) plan.md mentions PR-AUC, Confusion Matrix, and MAPE — these are missing from the evaluation; (4) no statistical significance testing or confidence intervals.

---

## Phase 8: Streamlit App & Bonus Extensions

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 7 |
| 3. Storage of Relevant Results | 7 |

**Feedback:**

**Streamlit App:** Well-structured with 5 tabs (Explorer, Discovery, Performance, Live Prediction, Conformance). Tab-per-file architecture is clean. Performance tab shows eval results with variant comparison. Live prediction tab correctly builds feature vectors and shows per-instance SHAP. Conformance tab visualizes rule compliance.

**Bonus - Prescriptive:** Solid threshold-based policy with cost-benefit analysis. Literature-grounded (Di Francescomarino et al., 2017). Risk tier visualization saved.

**Bonus - Generative AI:** Simplistic Markov chain for synthetic trace generation. Functional but academically weak — a first-order Markov model cannot capture the complex temporal dependencies of the RTFM process. No evaluation of synthetic trace quality (fitness to original model, activity distribution comparison).

Issues: (1) App loads XGBoost but SHAP tab references RF models — mismatch after refactoring; (2) Live prediction SHAP uses `TreeExplainer` which may not work with all XGBoost versions in the same way as RF; (3) No caching strategy for SHAP computation (slow on every prediction); (4) Generative AI module is too simple for MSc level — would need at least a comparison with real data statistics.

---

## Gesamtaufbau & Stringenz (projektübergreifend)

| Criterion | Score (1–10) |
|---|---|
| 4. Aufbau & Stringenz der Pipeline | 7 |

**Feedback:**

**Strengths:**
- Clear linear pipeline: Load → Describe → Clean → Discover → Conform → Engineer → Train → Evaluate → Interpret. Each step builds on the previous step's outputs.
- Data flows cleanly: `df_events.pkl` → `df_cleaned.pkl` → `prefix_k{k}_{variant}.parquet` → trained models → evaluation JSON.
- The `run_pipeline.py` orchestrator makes the end-to-end workflow reproducible with a single command.
- The CF/DA experimental design provides a clear scientific thread.
- `docs/` specs (leakage guardrails, prefix strategy, validation strategy, conformance rules) show thoughtful upfront planning.

**Weaknesses:**
- **Naming inconsistency:** Task numbers in print statements don't match phase numbers (t2 prints "TASK 3", t3 prints "TASK 2"). The filenames follow one numbering, the runtime output another.
- **Missing planned infrastructure:** `configs/` YAML files, `tests/` directory, `docs/decisions.md` — all specified in plan.md but never created. This weakens the "rigor" claim.
- **Redundant/legacy artifacts:** `data/features/X_rf_k2.pkl`, `y_rf_k2.pkl`, `X_rf_k5.pkl`, etc. are legacy files alongside the proper parquet system — indicates incomplete cleanup of earlier iterations.
- **Model-app inconsistency:** After refactoring, the app loads XGBoost but SHAP interpretability generates plots for RF. The live prediction SHAP also references RF explainer logic in comments. This breaks coherence.
- **No automated validation:** No tests, no CI, no assertions verifying that pipeline outputs match expected schemas. For a project claiming reproducibility, this is a gap.
- **Bonus modules not in pipeline:** `bonus_prescriptive.py` and `bonus_generative_ai.py` are not called by `run_pipeline.py` — they're orphaned scripts that must be run manually.

---

## Summary

| Criterion | Average Score |
|---|---|
| 1. Task Fulfillment | 7.6 |
| 2. Code Quality & Appropriateness | 7.0 |
| 3. Storage of Relevant Results | 7.7 |
| 4. Aufbau & Stringenz | 7.0 |

**Overall Grade: 7.3 / 10**

### Top 3 Strengths
1. **Conformance Checking** — Best module. Domain-grounded rules with literature citations, clean implementation, complete persistence.
2. **Feature Engineering & Leakage Prevention** — Teinemaa (2019) framework correctly implemented with temporal split debiasing, prefix-bounded aggregates, outcome-activity exclusion.
3. **End-to-end pipeline** — Single-command execution with proper data flow between stages. Clear experimental design (CF vs DA, multiple k values).

### Top 3 Priorities to Fix
1. **Model-SHAP inconsistency** — App uses XGBoost but SHAP plots are generated for RF. Either regenerate SHAP for XGBoost or switch app back to RF. This is a coherence bug.
2. **Missing planned infrastructure** — Add at minimum: `configs/` with externalized hyperparameters, basic sanity-check tests, and a `docs/decisions.md` summarizing key choices. These were explicitly promised in plan.md.
3. **Overfitting assessment bug** — Train AUC < Test AUC (negative gap) is suspicious. Verify that train metrics are computed on the full training set after fitting, not on a validation subset or pre-fit data.
