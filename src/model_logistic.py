"""
Model 1: Logistic Regression with Elastic Net Penalty
Baseline model for binary classification with L1+L2 regularization.

Elastic Net combines:
  - L1 (Lasso): feature selection via sparse coefficients
  - L2 (Ridge): handles multicollinearity, stable optimization
  penalty = alpha * l1_ratio * |w| + 0.5 * alpha * (1 - l1_ratio) * ||w||^2
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from evaluation import evaluate_model, print_metrics, plot_reliability_diagram

warnings.filterwarnings('ignore')


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               output_dir: str,
                               class_weight: str = 'balanced') -> dict:
    """
    Train Logistic Regression with Elastic Net regularization.
    
    Hyperparameter justification:
    - C=0.01: regularization strength; cross-validated on validation set
    - l1_ratio=0.5: balanced Elastic Net (equal L1+L2 contribution)
    - class_weight='balanced': compensates for 1:150 imbalance by 
      assigning inverse-frequency weights to each class
    - max_iter=2000: sufficient for convergence with lbfgs solver
    - solver='saga': supports Elastic Net (l1_ratio != 0 or 1)
    """
    print("=" * 70)
    print("MODEL 1: LOGISTIC REGRESSION (ELASTIC NET)")
    print("=" * 70)

    # Grid search over C and l1_ratio with cross-validation
    print("\n[INFO] Searching optimal hyperparameters via 5-fold stratified CV...")
    
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_auprc = -np.inf
    best_params = {}
    results_grid = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for C in C_values:
        for l1_r in l1_ratios:
            model = LogisticRegression(
                penalty='elasticnet',
                C=C,
                l1_ratio=l1_r,
                solver='saga',
                class_weight=class_weight,
                max_iter=500,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            cv_scores = cross_validate(
                model, X_train, y_train,
                cv=skf,
                scoring='average_precision',
                n_jobs=-1
            )
            mean_auprc = cv_scores['test_score'].mean()
            results_grid.append({'C': C, 'l1_ratio': l1_r, 'auprc': mean_auprc})
            
            if mean_auprc > best_auprc:
                best_auprc = mean_auprc
                best_params = {'C': C, 'l1_ratio': l1_r}

    print(f"\n  Best params: C={best_params['C']}, l1_ratio={best_params['l1_ratio']}")
    print(f"  Best CV AUPRC: {best_auprc:.4f}")

    # Train final model with best params
    final_model = LogisticRegression(
        penalty='elasticnet',
        C=best_params['C'],
        l1_ratio=best_params['l1_ratio'],
        solver='saga',
        class_weight=class_weight,
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)

    # Evaluate
    y_proba_val = final_model.predict_proba(X_val)[:, 1]
    y_proba_test = final_model.predict_proba(X_test)[:, 1]
    y_pred_test = final_model.predict(X_test)

    metrics_val = evaluate_model(y_val, final_model.predict(X_val), y_proba_val,
                                  MODEL_LOGISTIC + '_val')
    metrics_test = evaluate_model(y_test, y_pred_test, y_proba_test, MODEL_LOGISTIC)
    print_metrics(metrics_test)

    # Calibration check
    ece = plot_reliability_diagram(y_test, y_proba_test, 
                                   'Logistic Regression (ElasticNet)', output_dir)
    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
    if ece > 0.05:
        print("  ⚠️  Model is miscalibrated — Platt Scaling recommended")
    else:
        print("  ✓ Model is well-calibrated")

    # Feature importance (coefficients)
    coef = final_model.coef_[0]
    n_nonzero = np.sum(np.abs(coef) > 1e-6)
    print(f"\n  Non-zero coefficients: {n_nonzero}/{len(coef)} (L1 sparsity effect)")

    # Plot coefficients
    plot_coefficients(coef, output_dir)

    # Plot CV heatmap
    plot_cv_heatmap(results_grid, C_values, l1_ratios, best_params, output_dir)

    return {
        'model': final_model,
        'metrics': metrics_test,
        'y_proba': y_proba_test,
        'best_params': best_params,
        'ece': ece
    }


def plot_coefficients(coef: np.ndarray, output_dir: str, top_n: int = 25):
    """Plot top feature coefficients."""
    # Feature names (V1-V28 + engineered)
    n_v = 28
    feature_names = [f'V{i}' for i in range(1, n_v+1)]
    extra = ['Amount_log', 'Amount_sqrt', 'Hour', 'Hour_sin', 'Hour_cos',
             'Amount_bin_mean', 'Amount_bin_std', 'Amount_zscore_local',
             'V_norm', 'V_mean', 'V_std', 'V4_V11', 'V14_V17', 'V3_V10']
    feature_names = feature_names + extra
    # Trim to actual length
    feature_names = feature_names[:len(coef)]

    idx = np.argsort(np.abs(coef))[-top_n:]
    top_coef = coef[idx]
    top_names = [feature_names[i] if i < len(feature_names) else f'feat_{i}' 
                 for i in idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e74c3c' if c > 0 else '#3498db' for c in top_coef]
    ax.barh(range(top_n), top_coef, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient Value (ElasticNet)', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Coefficients\n(Red=Fraud indicator, Blue=Normal indicator)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logistic_coefficients.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Coefficient plot saved")


def plot_cv_heatmap(results_grid: list, C_values: list, l1_ratios: list,
                    best_params: dict, output_dir: str):
    """Plot CV AUPRC heatmap for hyperparameter grid."""
    matrix = np.zeros((len(C_values), len(l1_ratios)))
    for r in results_grid:
        i = C_values.index(r['C'])
        j = l1_ratios.index(r['l1_ratio'])
        matrix[i, j] = r['auprc']

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='CV AUPRC')
    ax.set_xticks(range(len(l1_ratios)))
    ax.set_xticklabels([str(l) for l in l1_ratios])
    ax.set_yticks(range(len(C_values)))
    ax.set_yticklabels([str(c) for c in C_values])
    ax.set_xlabel('l1_ratio', fontsize=12)
    ax.set_ylabel('C (regularization)', fontsize=12)
    ax.set_title('ElasticNet Hyperparameter CV-AUPRC Grid', fontsize=13, fontweight='bold')

    # Mark best
    best_i = C_values.index(best_params['C'])
    best_j = l1_ratios.index(best_params['l1_ratio'])
    ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                                fill=False, edgecolor='blue', linewidth=3))

    # Annotate
    for i in range(len(C_values)):
        for j in range(len(l1_ratios)):
            ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logistic_cv_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ CV hyperparameter heatmap saved")
