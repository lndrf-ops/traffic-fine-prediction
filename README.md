# 🚦 Predictive Process Analytics: Road Traffic Fines

End-to-end Process Mining and Predictive Analytics for the *Road Traffic Fine Management Process*. This project combines classical Process Mining with Machine Learning and Generative AI to predict payment defaults early.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Pipeline Tasks](#pipeline-tasks)
- [Team](#team)

## Overview

| Aspect | Detail |
|--------|--------|
| **Dataset** | Road Traffic Fine Management Process (150,370 cases, 561,470 events) |
| **Goal** | Binary classification: Payment vs. Send for Credit Collection |
| **Methods** | Logistic Regression, Random Forest, XGBoost, LSTM |
| **Frameworks** | pm4py, scikit-learn, XGBoost, PyTorch |

## Project Structure

```
traffic-fine-prediction/
├── data/
│   ├── raw/                        # Original .xes file (not in repo)
│   ├── cleaned/                    # Cleaned DataFrames (generated)
│   └── features/                   # Feature matrices (generated)
├── outputs/
│   ├── models/                     # Trained models (.pkl, .pth)
│   ├── plots/                      # Visualizations (.png)
│   └── reports/                    # JSON reports, synthetic event log
├── src/
│   ├── t1_data_loading.py          # Task 1: Read XES file
│   ├── t2_descriptive_analysis.py  # Task 2: Descriptive statistics & column profiling
│   ├── t3_data_cleaning.py         # Task 3: Cleaning & labeling
│   ├── t3b_batching_analysis.py    # Task 3.5: Batching analysis
│   ├── t4_process_discovery.py     # Task 4: Process Discovery & bottlenecks
│   ├── t5_conformance_checking.py  # Task 5: Conformance Checking
│   ├── t6_feature_engineering.py   # Task 6.1: Feature Engineering
│   ├── t6_train.py                 # Task 6.2: Model training
│   ├── t6_evaluate.py              # Task 6.3: Evaluation & overfitting analysis
│   ├── t6_interpretability.py      # Task 6.4: SHAP interpretability
│   ├── t7_prescriptive.py          # Task 7: Prescriptive Analytics
│   ├── t8_generative_ai.py         # Task 8: Synthetic event log generation
│   └── models.py                   # PyTorch LSTM architecture
├── app/
│   ├── app.py                      # Streamlit dashboard (entry point)
│   ├── theme.py                    # UI theming
│   └── tabs/                       # Dashboard tab modules
│       ├── tab1_explorer.py
│       ├── tab2_discovery.py
│       ├── tab3_performance.py
│       ├── tab4_predictive.py
│       ├── tab5_conformance.py
│       └── tab6_generative.py
├── run_pipeline.py                 # Pipeline orchestration
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python ≥ 3.10

### Installation

```bash
cd traffic-fine-prediction
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The dataset (`Road_Traffic_Fine_Management_Process.xes`) is included in `data/raw/`.

## Usage

### 1. Run the full pipeline

```bash
source venv/bin/activate
python run_pipeline.py
```

This executes all 12 tasks sequentially (~5–10 min) and generates outputs in `data/` and `outputs/`.

### 2. Start the Streamlit dashboard

```bash
streamlit run app/app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

> **Note:** The app requires the pipeline to have been run at least once (it reads from `outputs/` and `data/features/`).

### Run specific tasks

```bash
python run_pipeline.py --only 4      # Only Process Discovery
python run_pipeline.py --from 6.1    # From feature engineering onwards
```

## Pipeline Tasks

| # | Task | Description | Output |
|---|------|-------------|--------|
| 1 | Data Loading | Read XES file | `data/cleaned/df_events.pkl` |
| 2 | Descriptive Analysis | Statistics, dotted chart | `outputs/reports/descriptive_analysis.json` |
| 3 | Data Cleaning | Remove duplicates, labeling | `data/cleaned/df_cleaned.pkl` |
| 3.5 | Batching Analysis | Batch-processing detection | `outputs/reports/batching_analysis.json` |
| 4 | Process Discovery | Petri net, bottlenecks, variants | `outputs/plots/petri_net.dot` |
| 5 | Conformance Checking | Token replay, compliance rules | `outputs/reports/conformance_results.json` |
| 6.1 | Feature Engineering | Prefix features (k=2,3,5), CF & DA variants | `data/features/` |
| 6.2 | Training | LR, RF, XGBoost, LSTM | `outputs/models/` |
| 6.3 | Evaluation | AUC, F1, overfitting analysis | `outputs/reports/evaluation_results.json` |
| 6.4 | Interpretability | SHAP values | `outputs/plots/shap_*.png` |
| 7 | Prescriptive Analytics | Risk tiers, action recommendations | `outputs/reports/prescriptive_recommendations.csv` |
| 8 | Generative AI | Markov chain event log synthesis | `outputs/reports/synthetic_event_log.csv` |

## Team

| Name |
|------|
| Lennard Ruf |
| Markus Schneele |
| Ali Hawash |
| Krzysztof Olesiak |

