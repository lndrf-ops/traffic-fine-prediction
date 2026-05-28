"""Task 6.3: Model Diagnostics and Evaluation
- Systematic model comparison (RF vs LR vs LSTM, k=2 vs k=5)
- Overfitting risk: Train/Test gap analysis
- Per-class evaluation: Precision, Recall, F1 per class
- Saves: outputs/reports/evaluation_results.json, outputs/plots/overfitting_analysis.png
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(model, X, y, model_name, k, random_state=42):
    """Evaluate a model: train metrics, test metrics, gap analysis"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Re-train on this split for fair comparison
    model.fit(X_train, y_train)

    # Train metrics
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    # Test metrics
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    def calc_metrics(y_true, y_pred, y_proba):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
        }

    train_metrics = calc_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = calc_metrics(y_test, y_test_pred, y_test_proba)
    gap = {k: test_metrics[k] - train_metrics[k] for k in train_metrics}

    # Per-class metrics (test set)
    prec_per_class, rec_per_class, f1_per_class, support = \
        precision_recall_fscore_support(y_test, y_test_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred)

    return {
        'model': model_name,
        'k': k,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'gap': gap,
        'per_class': {
            'payment': {'precision': prec_per_class[0], 'recall': rec_per_class[0], 'f1': f1_per_class[0], 'support': int(support[0])},
            'collection': {'precision': prec_per_class[1], 'recall': rec_per_class[1], 'f1': f1_per_class[1], 'support': int(support[1])},
        },
        'confusion_matrix': {'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])},
        'samples': {'train': len(y_train), 'test': len(y_test)},
    }


def print_evaluation(result):
    """Print formatted evaluation report"""
    print(f"\n  {'─'*60}")
    print(f"  {result['model']} (k={result['k']})")
    print(f"  {'─'*60}")
    print(f"  {'Metric':<12} {'Train':>8} {'Test':>8} {'Gap':>8} {'Status'}")
    print(f"  {'─'*50}")

    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        t = result['train_metrics'][metric]
        te = result['test_metrics'][metric]
        g = result['gap'][metric]
        status = "✓" if abs(g) < 0.02 else "⚠" if abs(g) < 0.05 else "✗"
        print(f"  {metric:<12} {t:>8.4f} {te:>8.4f} {g:>+8.4f} {status}")

    pc = result['per_class']
    print(f"\n  Per-Class (Test):")
    print(f"    Payment:    P={pc['payment']['precision']:.4f} R={pc['payment']['recall']:.4f} F1={pc['payment']['f1']:.4f}")
    print(f"    Collection: P={pc['collection']['precision']:.4f} R={pc['collection']['recall']:.4f} F1={pc['collection']['f1']:.4f}")

    max_gap = max(abs(v) for v in result['gap'].values())
    if max_gap < 0.02:
        print(f"  -> Overfitting Risk: LOW (max gap {max_gap:.4f})")
    elif max_gap < 0.05:
        print(f"  -> Overfitting Risk: MODERATE (max gap {max_gap:.4f})")
    else:
        print(f"  -> Overfitting Risk: HIGH (max gap {max_gap:.4f})")


def visualize_results(results, save_path='outputs/plots/evaluation_comparison.png'):
    """Visualize model comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Evaluation: Train vs Test', fontsize=14, fontweight='bold')

    colors = {'Random Forest': '#3498db', 'Logistic Regression': '#e74c3c'}

    # 1. Accuracy comparison
    ax = axes[0]
    labels = [f"{r['model'][:3]} k={r['k']}" for r in results]
    train_acc = [r['train_metrics']['accuracy'] for r in results]
    test_acc = [r['test_metrics']['accuracy'] for r in results]
    x = np.arange(len(labels))

    ax.bar(x - 0.2, train_acc, 0.35, label='Train', alpha=0.7, color='#3498db')
    ax.bar(x + 0.2, test_acc, 0.35, label='Test', alpha=0.7, color='#e74c3c')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy: Train vs Test')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0.75, 0.90])
    ax.grid(axis='y', alpha=0.3)

    # 2. F1 Score comparison
    ax = axes[1]
    train_f1 = [r['train_metrics']['f1_score'] for r in results]
    test_f1 = [r['test_metrics']['f1_score'] for r in results]

    ax.bar(x - 0.2, train_f1, 0.35, label='Train', alpha=0.7, color='#3498db')
    ax.bar(x + 0.2, test_f1, 0.35, label='Test', alpha=0.7, color='#e74c3c')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score: Train vs Test')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0.75, 0.90])
    ax.grid(axis='y', alpha=0.3)

    # 3. Gap analysis
    ax = axes[2]
    gaps = [r['gap']['accuracy'] for r in results]
    bar_colors = ['green' if abs(g) < 0.02 else 'orange' if abs(g) < 0.05 else 'red' for g in gaps]

    ax.bar(x, gaps, color=bar_colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Gap (Test - Train)')
    ax.set_title('Accuracy Gap (Overfitting Indicator)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for i, (bar_x, val) in enumerate(zip(x, gaps)):
        ax.text(bar_x, val + 0.001, f'{val:+.4f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("TASK 6.3: Model Evaluation & Overfitting Analysis")
    print("=" * 60)

    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)

    all_results = []

    # Evaluate RF and LR for k=2 and k=5
    for k in [2, 5]:
        X = pd.read_pickle(f"data/features/X_rf_k{k}.pkl")
        y = pd.read_pickle(f"data/features/y_rf_k{k}.pkl")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        result_rf = evaluate_model(rf, X, y, 'Random Forest', k)
        all_results.append(result_rf)
        print_evaluation(result_rf)

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        result_lr = evaluate_model(lr, X, y, 'Logistic Regression', k)
        all_results.append(result_lr)
        print_evaluation(result_lr)

    # Visualize
    visualize_results(all_results)
    print(f"\n  ✅ Plot saved: outputs/plots/evaluation_comparison.png")

    # Save JSON report
    with open('outputs/reports/evaluation_results.json', 'w') as f:
        json.dump({'evaluations': all_results}, f, indent=2)
    print(f"  ✅ Report saved: outputs/reports/evaluation_results.json")

    # Summary Table
    print(f"\n  {'='*60}")
    print(f"  SUMMARY")
    print(f"  {'='*60}")
    print(f"  {'Model':<22} {'k':<4} {'Test Acc':<10} {'Test F1':<10} {'Max Gap':<10} {'Risk'}")
    print(f"  {'─'*60}")
    for r in all_results:
        max_gap = max(abs(v) for v in r['gap'].values())
        risk = 'Low' if max_gap < 0.02 else 'Moderate' if max_gap < 0.05 else 'High'
        print(f"  {r['model']:<22} {r['k']:<4} {r['test_metrics']['accuracy']:<10.4f} {r['test_metrics']['f1_score']:<10.4f} {max_gap:<10.4f} {risk}")


if __name__ == "__main__":
    main()
