# Project Review — Road Traffic Fine Management Process

## Review Checklist
- [x] Phase 0: Setup & Project Structure
- [x] Phase 1: Data Loading
- [x] Phase 2: Descriptive Analysis
- [x] Phase 3: Data Cleaning
- [x] Phase 3b: Batching Analysis
- [x] Phase 4: Process Discovery
- [x] Phase 5: Conformance Checking
- [x] Phase 6.1: Feature Engineering
- [x] Phase 6.2: Model Training
- [x] Phase 6.3: Model Evaluation
- [x] Phase 6.4: Interpretability (SHAP)
- [x] Phase 7: Prescriptive Analytics (Bonus)
- [x] Phase 8: Generative AI (Bonus)
- [x] Streamlit App
- [x] Gesamtaufbau & Stringenz reviewed

---

## Phase 0: Setup & Project Structure

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 9 |
| 3. Storage of Relevant Results | 8 |

**Feedback:** Excellent project scaffold. `CLAUDE.md` is detailed and well-organized — it documents coding conventions, domain decisions, model lineup, and spec pointers. The folder structure (`data/raw|cleaned|features`, `outputs/models|plots|reports`, `src/`) is logical and consistent. `run_pipeline.py` with `--from` and `--only` flags is a nice touch. Minor gap: no `docs/` methodology specs are actually present besides `references.md` — the CLAUDE.md references files like `docs/validation_strategy.md` and `docs/leakage_guardrails.md` that don't exist. This means key design rationale is scattered only in code comments rather than having a single reference doc.

---

## Phase 1: Data Loading

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 9 |
| 3. Storage of Relevant Results | 8 |

**Feedback:** Simple, clean, does what it needs to — and for a data-loading task, simplicity *is* quality. Handles both `.xes` and `.xes.gz` paths gracefully. Prints case/event/activity counts as a sanity check. Saves as pickle for downstream consumption. Minor note: `data/cleaned/df_events.pkl` is the parsed-but-not-yet-cleaned DataFrame living in the `cleaned/` directory, which is slightly confusing semantically — but acceptable since `data/cleaned/` effectively serves as the "derived data" folder (as opposed to the original XES source in `data/raw/`).

---

## Phase 2: Descriptive Analysis

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 8 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** Thorough column profiling with fill rates, min/max/median, unique counts. Correctly identifies low-quality columns, documents the known timestamp limitation. The dotted chart (500 cases) and activity frequency plot are appropriate visualizations. The structured JSON report (`descriptive_analysis.json`) is well-designed and machine-readable. Good that it runs on raw data *before* cleaning — this justifies the drop decisions in Phase 3. Minor: the `LOW_FILL_THRESHOLD = 1.0` could be parameterized or at least mentioned in CLAUDE.md.

---

## Phase 3: Data Cleaning

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 8 |
| 3. Storage of Relevant Results | 8 |

**Feedback:** Drops duplicates, removes identified low-quality columns, applies labeling logic with clear three-rule priority (documented in CLAUDE.md). The `determine_outcome` function has good comments explaining the precedence logic. Case-level attributes (`vehicleClass`, `article`, `points`) are correctly extracted from the first event. Saves both `df_cleaned.pkl` and `completed_cases.pkl` with clear semantics. Criticism: the rename of `concept:name` to `trace` via groupby is slightly confusing — a comment explains it, but the variable naming could be cleaner. The `event_position` column added at line 87 is computed *after* deduplication but *before* saving, which is correct but the ordering of operations (label logic → event_position → save) feels slightly disjointed.

---

## Phase 3b: Batching Analysis

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 9 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** This is one of the strongest modules. The observation-driven analysis (from dotted chart → batch hypothesis → quantification) demonstrates genuine analytical thinking. Using continuous metrics (activity_rate, events_per_active_day, max_median_ratio) instead of arbitrary thresholds is methodologically sound and explicitly justified. The waiting-time analysis and implication for MAE floor is insightful. The reference to Martin et al. (2017) on this exact dataset shows domain awareness. The resource attribution analysis (automated vs. manual activities) adds genuine depth. Code is clean, well-commented, uses logging appropriately. The two-panel plot is informative.

---

## Phase 4: Process Discovery

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 7 |
| 3. Storage of Relevant Results | 8 |

**Feedback:** Covers bottleneck analysis (mean + median), case duration distributions, variant analysis (top 5), performance spectrum, and Petri net discovery. The breadth is appropriate. Criticisms: (1) The performance spectrum is ad-hoc (manual plotting with matplotlib) rather than using pm4py's built-in performance spectrum — this works but loses the formal semantics. (2) The bottleneck analysis uses transition times but doesn't filter by count — a transition with N=2 and 500-day average isn't the same kind of bottleneck as N=50000. The `(N=...)` label partially addresses this but the analysis doesn't distinguish. (3) No report JSON is saved — all results are only in plots, making downstream programmatic use impossible. (4) The code is procedural (one big `main()`) without helper functions, making it harder to test or reuse individual analyses.

---

## Phase 5: Conformance Checking

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 9 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** Excellent. Five domain-specific rules derived from actual Italian traffic law (Codice della Strada), each with article references. The rules are non-trivial and demonstrate real domain understanding — not just generic process mining checks. Each function is cleanly separated, well-documented, handles edge cases (empty groups, missing activities). Token-based replay fitness provides the algorithmic conformance counterpart. All results saved in structured JSON with compliance rates, violation counts, and legal sources. One minor issue: the per-case iteration pattern (`for case_id, group in df_sorted.groupby(...)`) is O(n_cases) and could be vectorized for performance, but given the dataset size (~150k cases) it's acceptable.

---

## Phase 6.1: Feature Engineering

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 8 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** This is the most complex module and handles it well. The prefix-based approach (Teinemaa et al. 2019) is correctly implemented: leakage guardrails (removing outcome-revealing activities from prefixes), temporal split with debiasing (dropping boundary-crossing cases), two feature variants (CF vs DA). The `DA_PAYLOAD_COLS` selection with explicit drop justifications is thorough. LSTM sequence building with vocabulary encoding is correct. Saves everything needed: parquet feature files per (k, variant), sequences, split indices, activity vocab. Criticism: (1) The `build_prefixes` function is ~80 lines and does too many things — feature construction + target computation + leakage removal could be separated. (2) In `main()`, there's redundant code at lines 285–295 that re-computes the prefix filtering just to get `surviving_cases` for the split column — this is a code smell (the same logic is inside `build_prefixes`). (3) The article grouping (`ARTICLE_TOP_N`) is hardcoded rather than computed from data, which hurts reproducibility if the dataset changes.

---

## Phase 6.2: Model Training

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 8 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** All planned models are trained: baselines (Majority/Mean), linear (LogReg/LinReg), ensembles (RF/XGB), and LSTM for both tasks in both variants. Hyperparameter choices are commented with rationale ("no grid search — with 4 activities the models saturate quickly"). Class imbalance is handled (`class_weight="balanced"`, `scale_pos_weight`). Early stopping on validation set for XGB and LSTM. The macOS fork-safety fix (`OBJC_DISABLE_INITIALIZE_FORK_SAFETY`, `n_jobs=1`) shows practical awareness. LSTM architecture in `models.py` (shared source of truth) is appropriately simple (embedding → LSTM → dropout → linear). Criticism: (1) The LSTM trains on *all* prefix lengths simultaneously (pooled), which means it doesn't learn k-specific patterns — this is a design choice that should be more explicitly justified. (2) No hyperparameter tuning whatsoever is performed; even a brief Optuna/random-search experiment would strengthen the evaluation. (3) Feature columns are saved as JSON per model — good practice.

---

## Phase 6.3: Model Evaluation

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 8 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** Comprehensive metric set: AUC-ROC, accuracy, per-class F1/precision/recall for outcome; MAE and RMSE for remaining time. Evaluation is strictly test-set only ("never re-trains"). Overfitting assessment (train vs. test gap) is a valuable addition. Confusion matrices for XGBoost at k=5. Comparison plots. All results saved as JSON. The code correctly separates evaluation from training. Criticism: (1) The identical AUC scores for logreg/rf/xgb in the CF variant (visible in overfitting_assessment.json) strongly suggest a bug — likely all models converge to the same decision boundary because CF features (with only 4 non-outcome activity counts at k=2) have too few dimensions to differentiate models. This should be investigated and discussed. (2) No statistical significance testing (e.g., McNemar's test or bootstrap CIs) is performed to support claims about model differences.

---

## Phase 6.4: Interpretability (SHAP)

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 8 |
| 3. Storage of Relevant Results | 8 |

**Feedback:** SHAP TreeExplainer on XGBoost for both tasks, both variants, all k values — covers the full experiment matrix. Using `TreeExplainer` for exact Shapley values (not approximations) is the correct choice for tree ensembles. Generates summary dot plots (top 15 features). Also does remaining-time SHAP. Criticism: (1) Only XGBoost is explained — no SHAP for the LSTM (GradientExplainer or DeepExplainer would be appropriate). (2) Only global importance (summary plot), no local explanations (force plots for individual cases). (3) The `n_sample=200` is reasonable but could be larger for stable estimates. (4) No textual interpretation of the SHAP results is saved — the plots exist but no JSON summarizing which features matter most and why.

---

## Phase 7: Prescriptive Analytics (Bonus)

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 8 |
| 3. Storage of Relevant Results | 8 |

**Feedback:** A well-thought-out prescriptive module that goes beyond prediction to actionable recommendations. The three-tier policy (RED/YELLOW/GREEN) with cost-benefit analysis (expected net gain) is methodologically appropriate. References Di Francescomarino et al. (2017). The cost assumptions are clearly stated and labeled as "stylised for illustration." Saves CSV recommendations and a risk-tier scatter plot. Criticism: (1) Only uses XGBoost DA k=5 — would be stronger to show that recommendations are robust across model choices. (2) The cost-benefit model is purely static (no simulation of intervention effects). (3) Only 200 sampled cases — could easily run on the full test set. (4) The `partial effect` multiplier of 0.5 for reminders is arbitrary and undocumented.

---

## Phase 8: Generative AI (Bonus)

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 7 |
| 2. Code Quality & Appropriateness | 8 |
| 3. Storage of Relevant Results | 8 |

**Feedback:** Implements a first-order Markov chain for synthetic trace generation with a proper quality evaluation (Jensen-Shannon divergence, trace length comparison, directly-follows coverage). The limitation of first-order Markov is explicitly acknowledged in the docstring. The quality metrics are well-chosen and the trace-length histogram comparison is informative. Criticism: (1) A first-order Markov chain is *very* basic for a "Generative AI" task — it's essentially a frequency table. Even a second-order model or a simple RNN/GPT-style character model would be more interesting. (2) No conditional generation (e.g., generating traces for a specific outcome class). (3) The connection to the rest of the project is weak — the synthetic log isn't used anywhere downstream. (4) "Generative AI" in the title overpromises for what is effectively a statistical transition model.

---

## Streamlit App

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 8 |
| 3. Storage of Relevant Results | N/A |

**Feedback:** Six tabs covering exploration, discovery, performance evaluation, live prediction, conformance, and generative results. The live prediction tab (tab4) with constrained trace building (valid successors, requires-send-fine logic) is particularly well done — it prevents users from constructing impossible traces. The model comparison tables and overfitting assessment are well-presented. Code is modular (one file per tab). Criticism: (1) The app loads all models eagerly at startup via `@st.cache_resource` — with many models this could be slow. (2) Some tabs seem to duplicate logic from the source scripts rather than importing shared utilities. (3) No error handling if data files are missing (just `st.stop()`).

---

## Gesamtaufbau & Stringenz (projektübergreifend)

| Criterion | Score (1–10) |
|---|---|
| 4. Aufbau & Stringenz der Pipeline | 8 |

**Feedback:** The pipeline has a clear logical progression: Load → Describe → Clean → Discover → Conform → Engineer → Train → Evaluate → Interpret → Prescribe → Generate. Each step builds on the previous one's outputs (pickle → pickle → parquet → models → JSON reports). The two-variant experiment design (CF vs DA) runs as a consistent thread through feature engineering, training, evaluation, and interpretability — this is the project's strongest structural element.

**Strengths:**
- Data flows cleanly: `df_events.pkl` → `df_cleaned.pkl` + `completed_cases.pkl` → `prefix_k{k}_{variant}.parquet` → `outputs/models/` → `outputs/reports/`
- The CLAUDE.md serves as a single source of truth for design decisions
- Consistent naming (`t1_`, `t2_`, ...) and each module is independently runnable
- The CF vs DA comparison is maintained throughout — genuine scientific contribution

**Weaknesses:**
- `docs/` is nearly empty — the CLAUDE.md references specs that don't exist (`validation_strategy.md`, `leakage_guardrails.md`, etc.)
- The raw DataFrame is saved into `data/cleaned/` (semantic mismatch)
- Phase 4 (Process Discovery) doesn't save a structured report, breaking the otherwise consistent pattern
- The Batching Analysis (3b) is inserted between cleaning and discovery — logically it belongs *within* Phase 2 (Descriptive) or Phase 4 (Discovery), creating a minor ordering break
- The identical model scores in CF variant suggest an undiagnosed issue that should at minimum be acknowledged

---

## Summary

| Criterion | Average Score |
|---|---|
| 1. Task Fulfillment | 8.3 |
| 2. Code Quality & Appropriateness | 8.1 |
| 3. Storage of Relevant Results | 8.4 |
| 4. Aufbau & Stringenz | 8.0 |

**Overall grade: 8.2 / 10**

### Top 3 Strengths
1. **CF vs DA experiment design** — a genuine scientific thread running consistently from features through evaluation, providing a real comparative contribution
2. **Conformance Checking** — domain-specific rules grounded in actual Italian traffic law, not generic textbook checks
3. **Batching Analysis** — observation-driven, threshold-free, with proper literature references and clear implications for model limitations

### Top 3 Priorities to Fix
1. **Investigate identical CF variant scores** — logreg, RF, and XGBoost producing identical AUC-ROC values is almost certainly a bug (likely feature dimensionality collapse at low k with only activity counts)
2. **Create the missing `docs/` specs** — CLAUDE.md references `validation_strategy.md`, `leakage_guardrails.md`, `prefix_strategy.md` etc. that don't exist; either create them or remove the references
3. **Save a structured report for Phase 4 (Process Discovery)** — it's the only phase without a JSON output, breaking the consistent pattern that enables the Streamlit app to consume results programmatically
