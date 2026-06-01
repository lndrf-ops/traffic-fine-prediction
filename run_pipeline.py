"""
Traffic Fine Prediction - Main Pipeline
Executes all tasks sequentially: Data → Analysis → Cleaning → Discovery → Conformance → ML → Evaluation

Usage:
    python run_pipeline.py          # Run all tasks
    python run_pipeline.py --from 4 # Start from Task 4
    python run_pipeline.py --only 6 # Run only Task 6
"""

import sys
import time
from src import (
    t1_data_loading,
    t2_descriptive_analysis,
    t3_data_cleaning,
    t4_process_discovery,
    t5_conformance_checking,
    t6_feature_engineering,
    t6_train,
    t6_evaluate,
    t6_interpretability,
    bonus_prescriptive,
    bonus_generative_ai,
)


TASKS = [
    (1, "Data Loading", t1_data_loading),
    (2, "Descriptive Analysis", t2_descriptive_analysis),
    (3, "Data Cleaning", t3_data_cleaning),
    (4, "Process Discovery", t4_process_discovery),
    (5, "Conformance Checking", t5_conformance_checking),
    (6.1, "Feature Engineering", t6_feature_engineering),
    (6.2, "Model Training", t6_train),
    (6.3, "Model Evaluation", t6_evaluate),
    (6.4, "Interpretability (SHAP)", t6_interpretability),
    (7, "Bonus: Prescriptive Analytics", bonus_prescriptive),
    (8, "Bonus: Generative AI", bonus_generative_ai),
]


def run_pipeline(start_from=1, only=None):
    start_time = time.time()

    print("\n" + "═" * 60)
    print("🚀 TRAFFIC FINE PREDICTION PIPELINE")
    print("═" * 60)

    for task_id, task_name, module in TASKS:
        if only is not None and task_id != only:
            continue
        if task_id < start_from:
            continue

        print(f"\n▶️  TASK {task_id}: {task_name}")
        try:
            module.main()
        except Exception as e:
            print(f"\n❌ FEHLER in Task {task_id} ({task_name}):")
            print(f"   {e}")
            raise

    elapsed = (time.time() - start_time) / 60
    print("\n" + "═" * 60)
    print(f"✅ PIPELINE COMPLETED in {elapsed:.2f} minutes")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    start_from = 1
    only = None

    args = sys.argv[1:]
    if '--from' in args:
        idx = args.index('--from')
        start_from = float(args[idx + 1])
    if '--only' in args:
        idx = args.index('--only')
        only = float(args[idx + 1])

    run_pipeline(start_from=start_from, only=only)
