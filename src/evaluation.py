"""
Evaluation metrics and calibration utilities.
- F1-Macro, AUPRC, MCC
- Reliability Diagrams (calibration curves)
- Platt Scaling & Isotonic Regression calibration
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, average_precision_score, matthews_corrcoef,
    precision_recall_curve, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import warnings
import os
warnings.filterwarnings('ignore')


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   y_proba: np.ndarray, model_name: str,
                   threshold: float = 0.5) -> dict:
    """
    Compute all required metrics.
    
    Justification for metrics:
    - F1-Macro: balances precision/recall across both classes; crucial when
      class imbalance would make Accuracy misleading (99% accuracy by predicting
      all Normal).
    - AUPRC: area under Precision-Recall curve; more informative than ROC-AUC
      when positive class is rare, as it focuses on the minority class performance.
    - MCC: Matthews Correlation Coefficient; single scalar summarizing the 
      confusion matrix, unbiased even for extreme imbalances. Ranges [-1, 1].
    """
    y_pred_thresh = (y_proba >= threshold).astype(int)
    
    metrics = {
        'model': model_name,
        'threshold': threshold,
        'f1_macro': f1_score(y_true, y_pred_thresh, average='macro'),
        'f1_fraud': f1_score(y_true, y_pred_thresh, pos_label=1),
        'f1_normal': f1_score(y_true, y_pred_thresh, pos_label=0),
        'auprc': average_precision_score(y_true, y_proba),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'mcc': matthews_corrcoef(y_true, y_pred_thresh),
        'precision_fraud': 0,
        'recall_fraud': 0,
    }
    
    # Precision/Recall at threshold
    cm = confusion_matrix(y_true, y_pred_thresh)
    tn, fp, fn, tp = cm.ravel()
    metrics['precision_fraud'] = tp / (tp + fp + 1e-10)
    metrics['recall_fraud'] = tp / (tp + fn + 1e-10)
    metrics['specificity'] = tn / (tn + fp + 1e-10)
    metrics['confusion_matrix'] = cm

    return metrics


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    print(f"\n  {'─'*55}")
    print(f"  Model: {metrics['model']}")
    print(f"  {'─'*55}")
    print(f"  {'Metric':<30} {'Value':>10}")
    print(f"  {'─'*55}")
    print(f"  {'F1-Macro':<30} {metrics['f1_macro']:>10.4f}")
    print(f"  {'F1-Fraud (class 1)':<30} {metrics['f1_fraud']:>10.4f}")
    print(f"  {'AUPRC':<30} {metrics['auprc']:>10.4f}")
    print(f"  {'ROC-AUC':<30} {metrics['roc_auc']:>10.4f}")
    print(f"  {'MCC':<30} {metrics['mcc']:>10.4f}")
    print(f"  {'Precision (Fraud)':<30} {metrics['precision_fraud']:>10.4f}")
    print(f"  {'Recall (Fraud)':<30} {metrics['recall_fraud']:>10.4f}")
    print(f"  {'Specificity':<30} {metrics['specificity']:>10.4f}")
    
    cm = metrics['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"  {'':15s} Pred Normal  Pred Fraud")
    print(f"  {'True Normal':15s} {tn:>10,}  {fp:>10,}")
    print(f"  {'True Fraud':15s} {fn:>10,}  {tp:>10,}")


def plot_precision_recall_curves(results: list, output_dir: str, 
                                  y_true: np.ndarray, filename: str = 'pr_curves.png'):
    """Plot Precision-Recall curves for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    
    # PR Curve
    ax1 = axes[0]
    baseline = y_true.mean()
    ax1.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5,
                label=f'Random Baseline (AP={baseline:.4f})')
    
    for i, (model_name, y_proba, metrics) in enumerate(results):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auprc = metrics['auprc']
        ax1.plot(recall, precision, color=colors[i % len(colors)],
                 linewidth=2, label=f"{model_name} (AUPRC={auprc:.4f})")
    
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])

    # ROC Curve
    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Baseline')
    
    for i, (model_name, y_proba, metrics) in enumerate(results):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = metrics['roc_auc']
        ax2.plot(fpr, tpr, color=colors[i % len(colors)],
                 linewidth=2, label=f"{model_name} (AUC={roc_auc:.4f})")
    
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ PR and ROC curves saved")


def plot_reliability_diagram(y_true: np.ndarray, y_proba: np.ndarray,
                              model_name: str, output_dir: str,
                              n_bins: int = 10) -> float:
    """
    Plot Reliability Diagram (calibration curve).
    Returns Expected Calibration Error (ECE).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    
    # Compute ECE
    bin_sizes = []
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i+1])
        bin_sizes.append(mask.sum())
    ece = np.sum(np.abs(prob_true - prob_pred) * np.array(bin_sizes[:len(prob_true)]) / len(y_true))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Calibration curve
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    ax1.plot(prob_pred, prob_true, 's-', color='#e74c3c', linewidth=2,
             markersize=8, label=f'{model_name}\n(ECE={ece:.4f})')
    ax1.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title(f'Reliability Diagram — {model_name}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    # Histogram of predicted probabilities
    ax2 = axes[1]
    ax2.hist(y_proba[y_true == 0], bins=50, alpha=0.6, color='#3498db',
             label='Normal', density=True)
    ax2.hist(y_proba[y_true == 1], bins=50, alpha=0.7, color='#e74c3c',
             label='Fraud', density=True)
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Predicted Probability Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Calibration Analysis — {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('/', '_')
    plt.savefig(os.path.join(output_dir, f'calibration_{safe_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    return ece


def calibrate_model(estimator, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    method: str = 'sigmoid', cv: int = 5,
                    output_dir: str = '.', model_name: str = 'model') -> dict:
    """
    Apply calibration (Platt Scaling = 'sigmoid' or Isotonic Regression).
    
    - Platt Scaling: fits a sigmoid on top of model outputs.
      Works well for SVMs and other models with poor probability outputs.
    - Isotonic Regression: non-parametric monotone function.
      Better for large datasets; risk of overfitting on small sets.
    """
    print(f"\n  Calibrating with {method} method...")
    
    calibrated = CalibratedClassifierCV(estimator, cv=cv, method=method)
    calibrated.fit(X_train, y_train)
    y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
    
    ece_before = plot_reliability_diagram(y_test, 
                                           estimator.predict_proba(X_test)[:, 1] 
                                           if hasattr(estimator, 'predict_proba') 
                                           else estimator.decision_function(X_test),
                                           f'{model_name} (Before)', output_dir)
    ece_after = plot_reliability_diagram(y_test, y_proba_cal,
                                          f'{model_name} (After {method})', output_dir)
    
    print(f"  ECE before calibration: {ece_before:.4f}")
    print(f"  ECE after  calibration: {ece_after:.4f}")
    improvement = (ece_before - ece_after) / ece_before * 100
    if improvement > 0:
        print(f"  ✓ Calibration improved ECE by {improvement:.1f}%")
    else:
        print(f"  Model was already well-calibrated")

    return {
        'calibrated_model': calibrated,
        'y_proba_calibrated': y_proba_cal,
        'ece_before': ece_before,
        'ece_after': ece_after
    }


def compare_all_models(all_results: list, output_dir: str):
    """Create comparison table and radar chart of all models."""
    print(f"\n{'─'*60}")
    print("FINAL MODEL COMPARISON")
    print(f"{'─'*60}")

    # Table
    rows = []
    for m in all_results:
        rows.append({
            'Model': m['model'],
            'F1-Macro': f"{m['f1_macro']:.4f}",
            'AUPRC': f"{m['auprc']:.4f}",
            'MCC': f"{m['mcc']:.4f}",
            'ROC-AUC': f"{m['roc_auc']:.4f}",
            'F1-Fraud': f"{m['f1_fraud']:.4f}",
            'Recall-Fraud': f"{m['recall_fraud']:.4f}",
        })

    df_results = pd.DataFrame(rows)
    print(df_results.to_string(index=False))

    # Bar chart comparison
    metrics_to_plot = ['F1-Macro', 'AUPRC', 'MCC', 'ROC-AUC']
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 5))
    colors = ['#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    for ax, metric in zip(axes, metrics_to_plot):
        values = [float(r[metric]) for r in rows]
        model_names = [r['Model'] for r in rows]
        bars = ax.bar(range(len(model_names)), values, color=colors[:len(model_names)],
                      edgecolor='black', linewidth=0.7)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([n.replace('_', '\n') for n in model_names], fontsize=8)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.1])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Model Comparison — Key Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Model comparison chart saved")

    return df_results
