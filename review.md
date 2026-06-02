# Project Review — Road Traffic Fine Management Process

## Review Checklist
- [x] Phase 0: Setup & Project Structure
- [x] Phase 1: Data Loading
- [x] Phase 2: Descriptive Analysis
- [x] Phase 3: Data Cleaning
- [x] Phase 3b: Batching Analysis (Organizational)
- [x] Phase 4: Process Discovery
- [x] Phase 5: Conformance Checking
- [x] Phase 6: Feature Engineering + Training
- [x] Phase 7: Evaluation & Interpretability
- [x] Phase 8: Bonus (Prescriptive, Generative, Streamlit App)
- [x] Gesamtaufbau & Stringenz reviewed

---

## Phase 0: Setup & Project Structure

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 8 |
| 3. Storage of Relevant Results | 7 |

**Feedback:** Clean project structure with logical folder hierarchy (`src/`, `app/`, `data/`, `outputs/`). `run_pipeline.py` provides end-to-end execution with `--from` flag for partial re-runs. `PLAN.md` exists but is still marked "offen" for all phases — a living document would have been updated as implementation progressed. No `requirements.txt` lock file (only a basic `requirements.txt`). The `__init__.py` with module imports is good for the pipeline runner pattern. Minor: `PLAN.md` mentions `docs/` folder for specs but none exists.

---

## Phase 1: Data Loading

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 10 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** Exemplary KISS implementation (~45 lines). Loads XES via pm4py, converts to DataFrame, saves as pickle. Does exactly one thing, does it well. Clear docstring. Output persisted to `data/cleaned/df_events.pkl` for downstream consumption. Only minor gap: no validation/sanity check (e.g., asserting expected number of cases or columns).

---

## Phase 2: Descriptive Analysis

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 8 |
| 3. Storage of Relevant Results | 8 |

**Feedback:** Solid column profiling with fill rates, min/median/max, unique counts. Dotted chart generated. Design decision to run on pre-cleaning data is documented and sound. Output saved as JSON report + PNG. One concern: everything is in a single `main()` function (~130 lines) — extracting sub-functions would improve readability slightly, though the sequential nature makes it acceptable. The `LOW_FILL_THRESHOLD` constant is defined but the threshold-based recommendation is somewhat naive.

---

## Phase 3: Data Cleaning

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 9 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** Clean implementation: deduplication, column drops (with justification), completed-case filtering (Payment/Credit Collection only). The labeling logic — giving Credit Collection priority when both outcomes exist — is explicitly justified in comments. Outputs: `df_cleaned.pkl` (event-level) and `completed_cases.pkl` (case-level). Well-scoped, no over-engineering.

---

## Phase 3b: Batching Analysis (Organizational Perspective)

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 9 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** Strong analytical work. Uses continuous batch metrics (activity_rate, events_per_active_day, max/median ratio) per Martin et al. (2017) — no arbitrary thresholds. Identifies Send for Credit Collection as exclusive batching and Send Fine as spikey batching. The resource attribution analysis (9/11 automated) connects nicely to the conformance findings. Well-modularized functions, academic references throughout. Plot is informative (two-panel: concentration ranking + waiting time boxplots). JSON report saved. Note: not included in `run_pipeline.py` — must be run separately, which is inconsistent.

---

## Phase 4: Process Discovery

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 6 |
| 3. Storage of Relevant Results | 8 |

**Feedback:** Covers bottleneck analysis, case durations, variant analysis, performance spectrum, Petri net discovery, and organizational perspective — comprehensive scope. However, all logic lives in a single monolithic `main()` function (~155 lines) violating KISS through lack of decomposition. The org:resource check at the end references a column dropped in t3 — it will always print "not available," showing the code wasn't re-tested after the cleaning step. Multiple plots and artifacts saved correctly. The Petri net export (DOT + PNG) is a good touch.

---

## Phase 5: Conformance Checking

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 9 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** Excellent domain grounding — 5 rules derived from specific articles of the Italian Codice della Strada (Art. 201–204). Each rule is a separate function with legal citation in the docstring. Token-Based Replay adds the algorithmic conformance dimension. Results are concrete and interesting (52% violation on 90-day deadline is a standout finding). Only concern: the Python-loop iteration over 150k cases is O(n) per rule — acceptable for a university project but would need vectorization for production. Well-structured output JSON.

---

## Phase 6: Feature Engineering + Training

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 7 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** Feature engineering is well-designed: CF variant (activity indicators only) vs DA variant (+ duration, amount, points, vehicleClass, article). The top-10 article grouping and detailed drop-reason documentation are excellent. Temporal split (64/16/20) is correctly case-level. Training covers the full model zoo (majority/mean baselines, linear, RF, XGBoost, LSTM) × 2 variants × 3 prefix lengths = comprehensive. LSTM with early stopping on validation loss is correct.

Weaknesses: `t6_feature_engineering.py` (401 lines) is the most complex file — the `build_lstm_sequences()` function has duplicated prefix computation logic to recover case IDs, which is a code smell. The `import json as _json` inside function body in `t6_train.py` is unconventional. The `LSTMClassifier` class is duplicated between `t6_train.py` and `t6_evaluate.py` — should be in a shared module.

Storage is excellent: 60+ model files, feature column JSONs, parquet features — everything needed to reproduce.

---

## Phase 7: Evaluation & Interpretability

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 8 |
| 2. Code Quality & Appropriateness | 7 |
| 3. Storage of Relevant Results | 9 |

**Feedback:** Evaluation covers all planned metrics (F1 per-class, AUC-ROC, MAE, RMSE) with train/test overfitting assessment. Confusion matrices generated. SHAP interpretability for XGBoost is well-justified (TreeExplainer = exact Shapley values). The overfitting verdict system (ok/mild/overfit with gap thresholds) is a nice touch.

Weaknesses: `t6_evaluate.py` at 595 lines is the longest file and shows complexity creep — the LSTM evaluation section uses temporary dict keys (`_tmp`) which are hard to follow. `PREFIX_LENGTHS = [2, 3, 5, 8]` in `t6_interpretability.py` is stale (should be [2, 3, 5]) — will silently skip k=8 but the constant is misleading. The duplicated `LSTMClassifier` across files is the most significant design issue.

---

## Phase 8: Bonus Extensions & Streamlit App

| Criterion | Score (1–10) |
|---|---|
| 1. Task Fulfillment | 9 |
| 2. Code Quality & Appropriateness | 8 |
| 3. Storage of Relevant Results | 8 |

**Feedback:** Three bonus components delivered:

1. **Prescriptive Analytics** (`bonus_prescriptive.py`): Cost-benefit decision framework with documented assumptions, risk tiers, and actionable recommendations. Clean, well-scoped (~155 lines).

2. **Generative AI** (`bonus_generative_ai.py`): First-order Markov chain with honest JSD evaluation. Currently excluded from pipeline due to a data loading bug (concept:name key error) — should be fixed or the tab should show a clear "not available" state.

3. **Streamlit App**: 6-tab professional application with branded theming (Uni Leipzig), interactive live prediction with SHAP waterfall, conformance visualization, and model performance comparison. Tab4 (Live Prediction) is the highlight — DF-constrained trace builder with real-time XGBoost + SHAP explanation. The app is polished, well-cached, and handles missing data gracefully.

Minor: Generative AI not running end-to-end is a gap. The app's tab6 handles it with a warning, which is graceful but the underlying bug should be fixed.

---

## Gesamtaufbau & Stringenz (projektübergreifend)

| Criterion | Score (1–10) |
|---|---|
| 4. Aufbau & Stringenz der Pipeline | 8 |

**Feedback:**

**Strengths:**
- Clear red thread: Load → Explore → Clean → Discover → Conform → Features → Train → Evaluate → Interpret → App. Each step builds on prior outputs.
- Data flows logically: XES → df_events.pkl → df_cleaned.pkl → parquet features → models → evaluation JSON → app display.
- Consistent naming convention (`t1_`, `t2_`, ..., `bonus_`).
- The CF vs DA experiment design runs through the entire pipeline coherently.
- `run_pipeline.py` provides single-command reproducibility.

**Weaknesses:**
- `t3b_batching_analysis.py` is not registered in `run_pipeline.py` — an orphan that must be run manually.
- `bonus_generative_ai.py` was removed from the pipeline due to a bug — should be fixed or clearly documented as excluded.
- The `LSTMClassifier` duplication between train and evaluate breaks DRY and risks divergence.
- `t4_process_discovery.py` checks for `org:resource` which was already dropped in t3 — shows imperfect integration testing.
- `t6_interpretability.py` still references k=8 in its PREFIX_LENGTHS constant.
- `PLAN.md` was never updated from "offen" status — loses value as a living document.

**Overall:** The pipeline architecture is sound and well-scoped for an MSc project. The few inconsistencies (stale constants, one orphan script, one duplicated class) are minor and don't break functionality. The scientific narrative (CF vs DA, escalating model complexity, SHAP interpretability) is coherent throughout.

---

## Summary

| Criterion | Average Score |
|---|---|
| 1. Task Fulfillment | 8.6 |
| 2. Code Quality & Appropriateness | 8.0 |
| 3. Storage of Relevant Results | 8.5 |
| 4. Aufbau & Stringenz | 8.0 |

**Overall Grade: 8.3 / 10**

### Top 3 Strengths
1. **Domain-grounded conformance rules** — CdS legal citations elevate this beyond generic process mining
2. **Comprehensive experiment design** — CF vs DA × 5 model types × 3 prefix lengths with full evaluation + SHAP
3. **Professional Streamlit app** — Live prediction with DF-constrained trace builder and per-instance SHAP is impressive

### Top 3 Priorities to Fix
1. **Fix stale PREFIX_LENGTHS in `t6_interpretability.py`** — change [2,3,5,8] → [2,3,5] for consistency
2. **Extract shared `LSTMClassifier`** into `src/models.py` — eliminate duplication between train and evaluate
3. **Add `t3b_batching_analysis` to `run_pipeline.py`** and fix `bonus_generative_ai` data loading bug — ensure full pipeline runs end-to-end without manual intervention
