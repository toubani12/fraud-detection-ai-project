"""
Step 4: Interpretability using SHAP (SHapley Additive exPlanations)

SHAP values provide a game-theoretic approach to explain individual predictions.
Each feature's contribution to a prediction is computed as the average marginal
contribution across all possible feature orderings (Shapley values).

Advantages over alternative methods:
- Consistent: features with higher impact always get higher SHAP values
- Locally accurate: SHAP values sum exactly to the difference between the
  model output and the expected output
- Model-agnostic (KernelSHAP) or fast for tree models (TreeSHAP)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
import warnings
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

warnings.filterwarnings('ignore')


def get_feature_names(n_features: int) -> list:
    """Build feature names list matching the engineered feature set."""
    base = [f'V{i}' for i in range(1, 29)]
    extra = ['Amount_log', 'Amount_sqrt', 'Hour', 'Hour_sin', 'Hour_cos',
             'Amount_bin_mean', 'Amount_bin_std', 'Amount_zscore_local',
             'V_norm', 'V_mean', 'V_std', 'V4_V11', 'V14_V17', 'V3_V10']
    all_names = base + extra
    return all_names[:n_features]


def compute_shap_xgboost(model, X_test: np.ndarray, y_test: np.ndarray,
                          feature_names: list, output_dir: str,
                          n_sample: int = 2000):
    """
    Compute and visualize SHAP values for XGBoost using TreeSHAP.
    TreeSHAP is exact and efficient for tree-based models (O(TLD²) complexity).
    """
    print(f"\n  Computing TreeSHAP values (n={n_sample})...")
    
    # Subsample for speed
    np.random.seed(RANDOM_STATE)
    fraud_idx = np.where(y_test == 1)[0]
    normal_idx = np.random.choice(np.where(y_test == 0)[0],
                                   min(n_sample - len(fraud_idx), np.sum(y_test==0)),
                                   replace=False)
    sample_idx = np.concatenate([fraud_idx, normal_idx])[:n_sample]
    
    X_sample = X_test[sample_idx]
    y_sample = y_test[sample_idx]
    
    feature_names_clean = feature_names[:X_sample.shape[1]]
    
    # TreeSHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, shap_values may be list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        print(f"  ✓ SHAP values computed: {shap_values.shape}")
    except Exception as e:
        print(f"  ⚠️  TreeSHAP failed ({e}), falling back to Linear approximation")
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    # ── Plots ──────────────────────────────────────────────────────────
    
    # 1. Summary plot (beeswarm)
    print("  Generating SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names_clean,
        show=False, max_display=20
    )
    plt.title('SHAP Summary Plot — Feature Importance (XGBoost)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary_beeswarm.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ SHAP beeswarm summary saved")

    # 2. Bar plot (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 8))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    idx_sorted = np.argsort(mean_abs_shap)[-20:]
    
    ax.barh(range(20),
            mean_abs_shap[idx_sorted],
            color=plt.cm.YlOrRd(mean_abs_shap[idx_sorted] / mean_abs_shap[idx_sorted].max()),
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(20))
    ax.set_yticklabels([feature_names_clean[i] if i < len(feature_names_clean) else f'f{i}'
                        for i in idx_sorted], fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('SHAP Feature Importance (Top 20)\nMean Absolute SHAP Values',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_bar_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ SHAP bar importance saved")

    # 3. Dependence plots for top 3 features
    top3_idx = np.argsort(mean_abs_shap)[-3:]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, feat_idx in enumerate(top3_idx[::-1]):
        feat_name = feature_names_clean[feat_idx] if feat_idx < len(feature_names_clean) else f'f{feat_idx}'
        axes[i].scatter(X_sample[:, feat_idx], shap_values[:, feat_idx],
                        c=y_sample, cmap='RdBu_r', alpha=0.5, s=15)
        axes[i].set_xlabel(feat_name, fontsize=11)
        axes[i].set_ylabel('SHAP Value', fontsize=11)
        axes[i].set_title(f'SHAP Dependence: {feat_name}', fontsize=12, fontweight='bold')
        axes[i].axhline(y=0, color='black', linewidth=0.8, linestyle='--')
        axes[i].grid(True, alpha=0.3)
    fig.suptitle('SHAP Dependence Plots (Top 3 Features)\nRed=Fraud, Blue=Normal',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_dependence_top3.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ SHAP dependence plots saved")

    # 4. Waterfall plot for a specific fraud case
    fraud_local_idx = np.where(y_sample == 1)[0]
    if len(fraud_local_idx) > 0:
        # Pick the highest-probability fraud
        y_proba_sample = 1 / (1 + np.exp(-shap_values.sum(axis=1) + explainer.expected_value 
                                          if hasattr(explainer, 'expected_value') else 0))
        top_fraud = fraud_local_idx[0]
        
        print(f"  Generating waterfall plot for fraud case {sample_idx[top_fraud]}...")
        fig, ax = plt.subplots(figsize=(12, 7))
        
        shap_vals_single = shap_values[top_fraud]
        top_feat_idx = np.argsort(np.abs(shap_vals_single))[-15:]
        
        top_shap = shap_vals_single[top_feat_idx]
        top_feat_names = [feature_names_clean[i] if i < len(feature_names_clean) else f'f{i}'
                          for i in top_feat_idx]
        
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in top_shap]
        y_pos = range(len(top_shap))
        ax.barh(y_pos, top_shap, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_feat_names, fontsize=9)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP Value (impact on model output)', fontsize=12)
        ax.set_title('SHAP Waterfall — Individual Fraud Prediction\n'
                     'Red = pushes toward fraud, Blue = pushes toward normal',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_waterfall_fraud.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ SHAP waterfall (single fraud case) saved")

    # 5. Class-level SHAP: compare average SHAP for fraud vs normal
    fig, ax = plt.subplots(figsize=(12, 7))
    fraud_mask = y_sample == 1
    normal_mask = y_sample == 0
    
    top20_idx = np.argsort(mean_abs_shap)[-20:]
    
    if fraud_mask.sum() > 0 and normal_mask.sum() > 0:
        fraud_shap_mean = shap_values[fraud_mask][:, top20_idx].mean(axis=0)
        normal_shap_mean = shap_values[normal_mask][:, top20_idx].mean(axis=0)
        feat_labels = [feature_names_clean[i] if i < len(feature_names_clean) else f'f{i}'
                       for i in top20_idx]
        
        x = np.arange(20)
        width = 0.35
        ax.bar(x - width/2, fraud_shap_mean, width, label='Fraud', 
               color='#e74c3c', alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, normal_shap_mean, width, label='Normal',
               color='#3498db', alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean SHAP Value', fontsize=12)
        ax.set_title('Average SHAP Values by Class (Top 20 Features)\n'
                     'Positive SHAP = pushes prediction toward fraud',
                     fontsize=13, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_class_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ SHAP class comparison saved")

    return shap_values, mean_abs_shap


def compute_shap_logistic(model, X_test: np.ndarray, y_test: np.ndarray,
                            feature_names: list, output_dir: str,
                            n_sample: int = 1000):
    """Compute SHAP for Logistic Regression using LinearExplainer."""
    print(f"\n  Computing Linear SHAP values for Logistic Regression...")
    
    np.random.seed(RANDOM_STATE)
    sample_idx = np.random.choice(len(y_test), min(n_sample, len(y_test)), replace=False)
    X_sample = X_test[sample_idx]
    y_sample = y_test[sample_idx]
    
    feature_names_clean = feature_names[:X_sample.shape[1]]

    try:
        explainer = shap.LinearExplainer(model, X_sample, feature_perturbation='correlation_dependent')
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"  Fallback to KernelExplainer: {e}")
        background = shap.sample(X_sample, 100, random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values_raw = explainer.shap_values(X_sample[:200], nsamples=50)
        shap_values = shap_values_raw[1] if isinstance(shap_values_raw, list) else shap_values_raw

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    idx_sorted = np.argsort(mean_abs_shap)[-20:]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(20), mean_abs_shap[idx_sorted],
            color=plt.cm.Blues(mean_abs_shap[idx_sorted] / (mean_abs_shap[idx_sorted].max() + 1e-8)),
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(20))
    ax.set_yticklabels([feature_names_clean[i] if i < len(feature_names_clean) else f'f{i}'
                        for i in idx_sorted], fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('SHAP Feature Importance — Logistic Regression (Top 20)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_logistic_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ SHAP Logistic Regression importance saved")
    return shap_values, mean_abs_shap


def run_interpretability(models: dict, X_test: np.ndarray, y_test: np.ndarray,
                          feature_cols: list, output_dir: str):
    """Run full interpretability analysis."""
    print("=" * 70)
    print("STEP 4: INTERPRETABILITY — SHAP ANALYSIS")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    feature_names = list(feature_cols)

    results = {}

    # XGBoost SHAP
    if 'xgboost' in models:
        print("\n[XGBoost] TreeSHAP Analysis")
        shap_vals_xgb, mean_shap_xgb = compute_shap_xgboost(
            models['xgboost'], X_test, y_test, feature_names, output_dir
        )
        results['xgboost'] = {'shap_values': shap_vals_xgb, 'mean_abs': mean_shap_xgb}

    # Logistic Regression SHAP
    if 'logistic' in models:
        print("\n[Logistic Regression] Linear SHAP Analysis")
        shap_vals_lr, mean_shap_lr = compute_shap_logistic(
            models['logistic'], X_test, y_test, feature_names, output_dir
        )
        results['logistic'] = {'shap_values': shap_vals_lr, 'mean_abs': mean_shap_lr}

    print(f"\n{'='*70}")
    print("INTERPRETABILITY COMPLETE ✓")
    print(f"{'='*70}")

    return results
