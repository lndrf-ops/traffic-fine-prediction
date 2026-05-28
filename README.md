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
| **Methods** | Random Forest, Logistic Regression, LSTM |
| **Frameworks** | pm4py, scikit-learn, PyTorch |

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
│   ├── t2_data_cleaning.py         # Task 2: Cleaning & labeling
│   ├── t3_descriptive_analysis.py  # Task 3: Descriptive statistics
│   ├── t4_process_discovery.py     # Task 4: Process Discovery & bottlenecks
│   ├── t5_conformance_checking.py  # Task 5: Conformance Checking
│   ├── t6_feature_engineering.py   # Task 6.1: Feature Engineering
│   ├── t6_train.py                 # Task 6.2: Model training
│   ├── t6_evaluate.py              # Task 6.3: Evaluation & overfitting analysis
│   ├── t6_interpretability.py      # Task 6.4: SHAP interpretability
│   ├── bonus_prescriptive.py       # Bonus: Prescriptive Analytics
│   ├── bonus_generative_ai.py      # Bonus: Synthetic event log generation
│   └── models.py                   # PyTorch LSTM architecture
├── app/
│   └── app.py                      # Streamlit dashboard
├── run_pipeline.py                 # Pipeline orchestration
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python ≥ 3.10
- Place `Road_Traffic_Fine_Management_Process.xes` in `data/raw/`
  - Source: [4TU.ResearchData](https://data.4tu.nl/articles/dataset/Road_Traffic_Fine_Management_Process/12683249)

### Installation

```bash
git clone https://github.com/lndrf-ops/traffic-fine-prediction.git
cd traffic-fine-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

> **Note:** Always activate the virtual environment first: `source venv/bin/activate`

### Run full pipeline

```bash
python run_pipeline.py
```

### Run specific tasks

```bash
python run_pipeline.py --only 3      # Only descriptive analysis
python run_pipeline.py --from 6      # From feature engineering onwards
```

### Start Streamlit dashboard

```bash
streamlit run app/app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

## Pipeline Tasks

| # | Task | Description | Output |
|---|------|-------------|--------|
| 1 | Data Loading | Read XES file | `data/cleaned/df_events.pkl` |
| 2 | Data Cleaning | Remove duplicates, labeling | `data/cleaned/df_cleaned.pkl` |
| 3 | Descriptive Analysis | Statistics, dotted chart | `outputs/plots/dotted_chart.png` |
| 4 | Process Discovery | Petri net, bottlenecks, variants | `outputs/plots/petri_net.png` |
| 5 | Conformance Checking | Token replay, compliance rules | `outputs/reports/conformance.json` |
| 6.1 | Feature Engineering | k-gram features (k=2, k=5), LSTM sequences | `data/features/` |
| 6.2 | Training | RF, LR, LSTM | `outputs/models/` |
| 6.3 | Evaluation | Accuracy, overfitting analysis, per-class metrics | `outputs/plots/` |
| 6.4 | Interpretability | SHAP values | `outputs/plots/shap_*.png` |
| 7 | Generative AI | Markov chain event log synthesis | `outputs/reports/synthetic_event_log.csv` |

## Team

| Name |
|------|
| Lennard Ruf |
| Markus Schneele |
| Ali Hawash |
| Krzysztof Olesiak |

