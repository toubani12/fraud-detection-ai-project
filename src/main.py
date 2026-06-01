"""
Main Pipeline — Credit Card Fraud Detection Project
Orchestrates all 4 steps:
  1. EDA & Data Preparation
  2. Model Training (Logistic, Random Forest, XGBoost)
  3. Evaluation & Calibration
  4. Interpretability (SHAP)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from config import *
from eda import run_eda
from model_logistic import train_logistic_regression
from model_rf import train_random_forest
from model_xgboost import train_xgboost
from evaluation import (
    evaluate_model, print_metrics, plot_precision_recall_curves,
    plot_reliability_diagram, compare_all_models, calibrate_model
)
from interpretability import run_interpretability

import joblib


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║    CREDIT CARD FRAUD DETECTION — AI PROJECT                         ║
║    Classification Robuste en Environnement Déséquilibré             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Step 1: EDA & Feature Engineering                                  ║
║  Step 2: Model Development (LR, RF, XGBoost)                        ║
║  Step 3: Evaluation & Calibration                                   ║
║  Step 4: Interpretability (SHAP)                                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def save_results_summary(all_metrics: list, output_dir: str):
    """Save final results to CSV and markdown."""
    rows = []
    for m in all_metrics:
        rows.append({
            'Model': m['model'],
            'F1_Macro': round(m['f1_macro'], 4),
            'F1_Fraud': round(m['f1_fraud'], 4),
            'AUPRC': round(m['auprc'], 4),
            'ROC_AUC': round(m['roc_auc'], 4),
            'MCC': round(m['mcc'], 4),
            'Precision_Fraud': round(m['precision_fraud'], 4),
            'Recall_Fraud': round(m['recall_fraud'], 4),
            'Specificity': round(m['specificity'], 4),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)
    
    # Markdown table
    md_lines = [
        "# Model Comparison Results\n",
        "## Metric Justification\n",
        "- **F1-Macro**: Harmonic mean of precision and recall, averaged across both classes.",
        "  Critical because Accuracy is deceptive for 1:150 imbalanced datasets (trivially 99.3% by predicting all normal).\n",
        "- **AUPRC**: Area Under Precision-Recall Curve. More informative than ROC-AUC",
        "  for rare positive classes, as it directly measures ability to find fraud without false alarms.\n",
        "- **MCC**: Matthews Correlation Coefficient. Single balanced metric from all 4",
        "  confusion matrix cells. Range [-1, 1]; 0 = random, 1 = perfect. Unbiased under imbalance.\n",
        "\n## Results Table\n",
        df.to_markdown(index=False) if hasattr(df, 'to_markdown') else df.to_string(),
        "\n"
    ]
    with open(os.path.join(output_dir, 'results_summary.md'), 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n  ✓ Results saved to {output_dir}/results_summary.csv")
    return df


def run_full_pipeline():
    """Execute the complete ML pipeline."""
    start_time = time.time()
    print_banner()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: EDA & Data Preparation
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'█'*70}")
    print("█ STEP 1: EDA & DATA PREPARATION")
    print(f"{'█'*70}\n")
    
    eda_data = run_eda(OUTPUT_DIR)
    
    X_train = eda_data['X_train']
    X_val   = eda_data['X_val']
    X_test  = eda_data['X_test']
    y_train = eda_data['y_train']
    y_val   = eda_data['y_val']
    y_test  = eda_data['y_test']
    feature_cols = eda_data['feature_cols']

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: Model Development
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'█'*70}")
    print("█ STEP 2: MODEL DEVELOPMENT")
    print(f"{'█'*70}")

    # 2.1 Logistic Regression (ElasticNet)
    lr_results = train_logistic_regression(
        X_train, y_train, X_val, y_val, X_test, y_test, OUTPUT_DIR
    )

    # 2.2 Random Forest with Proximity
    rf_results = train_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test, OUTPUT_DIR
    )

    # 2.3 XGBoost Cost-Sensitive + Bayesian Optimization
    xgb_results = train_xgboost(
        X_train, y_train, X_val, y_val, X_test, y_test, OUTPUT_DIR
    )

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: Evaluation & Calibration
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'█'*70}")
    print("█ STEP 3: EVALUATION & CALIBRATION")
    print(f"{'█'*70}\n")

    all_metrics = [
        lr_results['metrics'],
        rf_results['metrics'],
        xgb_results['metrics'],
    ]
    
    # Add SPW vs Focal metrics
    if 'metrics_spw' in xgb_results:
        all_metrics.extend([
            xgb_results['metrics_spw'],
            xgb_results['metrics_focal']
        ])

    # Unified PR/ROC curves
    pr_results = [
        (MODEL_LOGISTIC.replace('_', ' '), lr_results['y_proba'], lr_results['metrics']),
        (MODEL_RF.replace('_', ' '), rf_results['y_proba'], rf_results['metrics']),
        ('XGBoost SPW', xgb_results['y_proba_spw'], xgb_results['metrics_spw']),
        ('XGBoost Focal', xgb_results['y_proba_focal'], xgb_results['metrics_focal']),
    ]
    plot_precision_recall_curves(pr_results, OUTPUT_DIR, y_test)

    # Calibration comparison
    print("\n  Calibration Analysis:")
    ecces = {
        'Logistic': lr_results['ece'],
        'Random Forest': rf_results['ece'],
        'XGBoost': xgb_results['ece'],
    }
    
    print("\n  Model Calibration Summary (ECE):")
    for name, ece in ecces.items():
        status = "✓ well-calibrated" if ece < 0.05 else "⚠️  needs calibration"
        print(f"    {name:20s}: ECE={ece:.4f}  {status}")

    # Apply Platt Scaling to worst calibrated model
    worst_model_name = max(ecces, key=ecces.get)
    if ecces[worst_model_name] > 0.05:
        print(f"\n  Applying Platt Scaling to {worst_model_name}...")
        model_map = {'Logistic': lr_results, 'Random Forest': rf_results, 'XGBoost': xgb_results}
        cal_result = calibrate_model(
            model_map[worst_model_name]['model'],
            X_train, y_train, X_test, y_test,
            method='sigmoid', cv=3,
            output_dir=OUTPUT_DIR,
            model_name=worst_model_name
        )

    # Compare all models
    df_results = compare_all_models(all_metrics, OUTPUT_DIR)
    save_results_summary(all_metrics[:3], OUTPUT_DIR)

    # ──────────────────────────────────────────────────────────────────
    # STEP 4: Interpretability
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'█'*70}")
    print("█ STEP 4: INTERPRETABILITY (SHAP)")
    print(f"{'█'*70}")

    models_for_shap = {
        'xgboost': xgb_results['model'],
        'logistic': lr_results['model'],
    }
    
    shap_results = run_interpretability(
        models_for_shap, X_test, y_test, feature_cols, OUTPUT_DIR
    )

    # ──────────────────────────────────────────────────────────────────
    # Save Models
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("SAVING TRAINED MODELS")
    print(f"{'─'*50}")
    
    joblib.dump(lr_results['model'], os.path.join(OUTPUT_DIR, 'model_logistic.pkl'))
    joblib.dump(rf_results['model'], os.path.join(OUTPUT_DIR, 'model_rf.pkl'))
    joblib.dump(xgb_results['model'], os.path.join(OUTPUT_DIR, 'model_xgboost.pkl'))
    joblib.dump(eda_data['scaler'], os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    print("  ✓ Models saved to outputs/ directory")

    # ──────────────────────────────────────────────────────────────────
    # Final Summary
    # ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'═'*70}")
    print("PIPELINE COMPLETE")
    print(f"{'═'*70}")
    print(f"  Total execution time: {elapsed/60:.1f} minutes")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"\n  Generated outputs:")
    
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        fsize = os.path.getsize(fpath) / 1024
        print(f"    {f:<45} {fsize:>8.1f} KB")
    
    print(f"\n{'═'*70}")
    print("  Best model by AUPRC:")
    best_row = df_results.loc[df_results['AUPRC'].astype(float).idxmax()]
    print(f"    {best_row['Model']} — AUPRC={best_row['AUPRC']}, MCC={best_row['MCC']}")
    print(f"{'═'*70}\n")
    
    return {
        'eda': eda_data,
        'lr': lr_results,
        'rf': rf_results,
        'xgb': xgb_results,
        'shap': shap_results,
        'results_df': df_results
    }


if __name__ == '__main__':
    results = run_full_pipeline()
