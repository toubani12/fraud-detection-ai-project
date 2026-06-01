"""
Model 3: XGBoost with Cost-Sensitive Learning and Bayesian Optimization (Optuna)

Two cost-sensitive strategies compared:
  A) scale_pos_weight: scales gradient contributions for positive class
  B) Custom asymmetric loss function (focal loss variant)

Hyperparameter optimization via Optuna (TPE sampler).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score
import warnings
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from evaluation import evaluate_model, print_metrics, plot_reliability_diagram

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Custom Focal Loss ─────────────────────────────────────────────────────────
def focal_loss_gradient(y_pred: np.ndarray, y_true: np.ndarray,
                         gamma: float = 2.0, alpha: float = 0.75):
    """
    Focal Loss gradient and hessian for XGBoost custom objective.
    
    FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    Motivation: Down-weights easy examples (normal transactions that the model 
    already classifies correctly) and focuses learning on hard examples 
    (borderline frauds). The gamma parameter controls how strongly easy 
    examples are down-weighted.
    
    - gamma=0: equivalent to cross-entropy
    - gamma=2: standard focal loss (Lin et al., 2017)
    - alpha=0.75: higher weight for positive (fraud) class
    """
    # Sigmoid of raw predictions
    p = 1.0 / (1.0 + np.exp(-y_pred))
    
    # For positive class
    pos_term = alpha * np.power(1 - p, gamma) * (gamma * p * np.log(p + 1e-7) - (1 - p))
    # For negative class
    neg_term = (1 - alpha) * np.power(p, gamma) * ((1 + gamma * (1 - p) * np.log(1 - p + 1e-7)) * p - gamma * (1-p) * p)
    
    grad = np.where(y_true == 1, pos_term, -neg_term)
    # Approximate hessian (use upper bound for stability)
    hess = np.ones_like(grad) * 0.5

    return grad, hess


class FocalLossObjective:
    """Wrapper for XGBoost custom objective."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y_true = dtrain.get_label()
        grad, hess = focal_loss_gradient(y_pred, y_true, self.gamma, self.alpha)
        return grad, hess


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ── Strategy A: scale_pos_weight ─────────────────────────────────────────────
def optimize_xgboost_spw(X_train, y_train, X_val, y_val, output_dir: str) -> dict:
    """
    Bayesian hyperparameter optimization for XGBoost with scale_pos_weight.
    
    Hyperparameter search space justification:
    - max_depth [3, 10]: shallow trees (3) prevent overfitting; deep trees (10)
      capture complex fraud patterns; optimal usually 4-7 for tabular data.
    - learning_rate [0.01, 0.3]: lower lr needs more trees but generalizes better;
      higher lr converges fast but may overfit.
    - n_estimators [100, 600]: paired with learning_rate; more trees at lower lr.
    - subsample [0.6, 1.0]: row subsampling reduces variance (like bootstrap).
    - colsample_bytree [0.6, 1.0]: feature subsampling per tree (like RF).
    - gamma [0, 5]: minimum loss reduction for a split; acts as regularization.
    - lambda (reg_lambda) [1, 10]: L2 regularization on leaf weights.
    - alpha (reg_alpha) [0, 5]: L1 regularization; promotes sparse feature use.
    - min_child_weight [1, 10]: minimum sum of hessian in a leaf; controls overfitting.
    - scale_pos_weight: ratio of negative to positive class count — standard 
      approach for imbalanced datasets; amplifies fraud gradient updates.
    """
    fraud_count = np.sum(y_train == 1)
    normal_count = np.sum(y_train == 0)
    default_spw = normal_count / fraud_count
    
    print(f"\n  Default scale_pos_weight = {default_spw:.1f} (N_normal / N_fraud)")

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 
                                                      default_spw * 0.5, 
                                                      default_spw * 2.0),
        }
        
        model = xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric='aucpr',
            random_state=RANDOM_STATE,
            tree_method='hist',
            n_jobs=-1
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        
        y_proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_proba)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name='xgboost_spw'
    )
    study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, n_jobs=1)
    
    return study


def optimize_xgboost_focal(X_train, y_train, X_val, y_val, output_dir: str) -> dict:
    """
    Bayesian optimization for XGBoost with custom Focal Loss.
    
    Additional hyperparameters:
    - gamma_focal [0.5, 5.0]: focal loss focusing parameter
    - alpha_focal [0.5, 0.95]: positive class weight in focal loss
    """
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma_focal': trial.suggest_float('gamma_focal', 0.5, 5.0),
            'alpha_focal': trial.suggest_float('alpha_focal', 0.5, 0.95),
        }
        
        focal_obj = FocalLossObjective(
            gamma=params.pop('gamma_focal'),
            alpha=params.pop('alpha_focal')
        )
        
        model = xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            tree_method='hist',
            n_jobs=-1
        )
        model.set_params(objective=focal_obj)
        
        try:
            model.fit(X_train, y_train, verbose=False)
            raw_scores = model.get_booster().predict(xgb.DMatrix(X_val), output_margin=True)
            y_proba = sigmoid(raw_scores)
            return average_precision_score(y_val, y_proba)
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name='xgboost_focal'
    )
    study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, n_jobs=1)
    
    return study


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   output_dir: str) -> dict:
    """
    Full XGBoost training pipeline with two cost-sensitive strategies.
    """
    print("=" * 70)
    print("MODEL 3: XGBOOST — COST-SENSITIVE + BAYESIAN OPTIMIZATION")
    print("=" * 70)

    fraud_count = np.sum(y_train == 1)
    normal_count = np.sum(y_train == 0)
    default_spw = normal_count / fraud_count

    # ── Strategy A: scale_pos_weight ──────────────────────────────────
    print("\n[STRATEGY A] scale_pos_weight — Bayesian Optimization...")
    print(f"  Running {N_TRIALS} Optuna trials...")
    
    study_spw = optimize_xgboost_spw(X_train, y_train, X_val, y_val, output_dir)
    best_spw = study_spw.best_params
    
    print(f"\n  ✓ Best params (SPW): {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in best_spw.items()}, indent=4)}")
    print(f"  Best AUPRC (val): {study_spw.best_value:.4f}")
    
    # Train final SPW model
    model_spw = xgb.XGBClassifier(
        **best_spw,
        use_label_encoder=False,
        eval_metric='aucpr',
        random_state=RANDOM_STATE,
        tree_method='hist',
        n_jobs=-1
    )
    model_spw.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

    y_proba_spw = model_spw.predict_proba(X_test)[:, 1]
    metrics_spw = evaluate_model(y_test, model_spw.predict(X_test), 
                                  y_proba_spw, MODEL_XGBOOST + '_SPW')
    print_metrics(metrics_spw)

    # ── Strategy B: Focal Loss ─────────────────────────────────────────
    print("\n[STRATEGY B] Custom Focal Loss — Bayesian Optimization...")
    print(f"  Running {N_TRIALS} Optuna trials...")
    
    study_focal = optimize_xgboost_focal(X_train, y_train, X_val, y_val, output_dir)
    best_focal_params = study_focal.best_params

    print(f"\n  ✓ Best params (Focal): AUPRC = {study_focal.best_value:.4f}")

    # Train final Focal model
    gamma_focal = best_focal_params.pop('gamma_focal', 2.0)
    alpha_focal = best_focal_params.pop('alpha_focal', 0.75)
    focal_obj = FocalLossObjective(gamma=gamma_focal, alpha=alpha_focal)

    model_focal = xgb.XGBClassifier(
        **best_focal_params,
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        tree_method='hist',
        n_jobs=-1
    )
    model_focal.set_params(objective=focal_obj)
    model_focal.fit(X_train, y_train, verbose=False)

    # Get probabilities from focal model
    raw_scores = model_focal.get_booster().predict(
        xgb.DMatrix(X_test), output_margin=True
    )
    y_proba_focal = sigmoid(raw_scores)
    metrics_focal = evaluate_model(y_test, (y_proba_focal >= 0.5).astype(int),
                                    y_proba_focal, MODEL_XGBOOST + '_Focal')
    print_metrics(metrics_focal)

    # ── Choose best model ─────────────────────────────────────────────
    if metrics_spw['auprc'] >= metrics_focal['auprc']:
        print(f"\n  ✓ scale_pos_weight strategy wins (AUPRC: {metrics_spw['auprc']:.4f} vs {metrics_focal['auprc']:.4f})")
        best_model = model_spw
        best_metrics = metrics_spw
        best_proba = y_proba_spw
        best_strategy = 'scale_pos_weight'
    else:
        print(f"\n  ✓ Focal Loss strategy wins (AUPRC: {metrics_focal['auprc']:.4f} vs {metrics_spw['auprc']:.4f})")
        best_model = model_focal
        best_metrics = metrics_focal
        best_proba = y_proba_focal
        best_strategy = 'focal_loss'

    # Calibration
    ece = plot_reliability_diagram(y_test, best_proba, 'XGBoost (Best Strategy)', output_dir)
    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")

    # Optuna convergence plots
    plot_optuna_convergence(study_spw, study_focal, output_dir)

    # Feature importance
    plot_xgb_feature_importance(best_model, output_dir)

    # Strategy comparison
    plot_strategy_comparison(metrics_spw, metrics_focal, output_dir)

    return {
        'model': best_model,
        'model_spw': model_spw,
        'model_focal': model_focal,
        'metrics': best_metrics,
        'metrics_spw': metrics_spw,
        'metrics_focal': metrics_focal,
        'y_proba': best_proba,
        'y_proba_spw': y_proba_spw,
        'y_proba_focal': y_proba_focal,
        'study_spw': study_spw,
        'study_focal': study_focal,
        'best_strategy': best_strategy,
        'ece': ece
    }


def plot_optuna_convergence(study_spw, study_focal, output_dir: str):
    """Plot Optuna optimization history for both strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, study, title, color in [
        (axes[0], study_spw, 'scale_pos_weight Strategy', '#3498db'),
        (axes[1], study_focal, 'Focal Loss Strategy', '#e74c3c')
    ]:
        trials = study.trials
        trial_nums = [t.number for t in trials if t.value is not None]
        values = [t.value for t in trials if t.value is not None]
        
        if not values:
            ax.text(0.5, 0.5, 'No completed trials', transform=ax.transAxes, 
                    ha='center', va='center')
            continue
        
        # Running best
        running_best = [max(values[:i+1]) for i in range(len(values))]
        
        ax.scatter(trial_nums, values, alpha=0.4, s=20, color=color, label='Trial AUPRC')
        ax.plot(trial_nums, running_best, color='navy', linewidth=2.5,
                label='Best so far', zorder=5)
        ax.axhline(y=max(values), color='red', linestyle='--', linewidth=1.5,
                   label=f'Best: {max(values):.4f}')
        
        ax.set_xlabel('Trial Number', fontsize=12)
        ax.set_ylabel('Objective Value (AUPRC)', fontsize=12)
        ax.set_title(f'Optuna Convergence — {title}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([max(0, min(values) - 0.05), min(1, max(values) + 0.05)])

    fig.suptitle('Bayesian Optimization Convergence (TPE Sampler)\n'
                 'Demonstrates optimal exploration of hyperparameter space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optuna_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Optuna convergence plots saved")

    # Parameter importance
    try:
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
        for ax, study, title in [
            (axes2[0], study_spw, 'scale_pos_weight'),
            (axes2[1], study_focal, 'Focal Loss')
        ]:
            importances = optuna.importance.get_param_importances(study)
            if importances:
                params = list(importances.keys())[:10]
                imp_vals = [importances[p] for p in params]
                bars = ax.barh(params, imp_vals, 
                               color=plt.cm.viridis(np.array(imp_vals) / max(imp_vals)),
                               edgecolor='black')
                ax.set_xlabel('Relative Importance', fontsize=11)
                ax.set_title(f'Hyperparameter Importance — {title}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optuna_param_importance.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Parameter importance plots saved")
    except Exception as e:
        print(f"  ⚠️  Could not compute parameter importance: {e}")


def plot_xgb_feature_importance(model, output_dir: str, top_n: int = 25):
    """Plot XGBoost feature importances (gain)."""
    try:
        importance_dict = model.get_booster().get_score(importance_type='gain')
        if not importance_dict:
            return
        
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_items)
        importances = np.array(importances)
        importances = importances / importances.sum()

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(range(len(features)), importances,
                       color=plt.cm.YlOrRd(importances / importances.max()),
                       edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Normalized Gain Importance', fontsize=12)
        ax.set_title(f'XGBoost — Top {len(features)} Feature Importances (Gain)',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'xgb_feature_importance.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ XGBoost feature importance saved")
    except Exception as e:
        print(f"  ⚠️  Could not plot XGB feature importance: {e}")


def plot_strategy_comparison(metrics_spw: dict, metrics_focal: dict, output_dir: str):
    """Compare scale_pos_weight vs focal loss strategies."""
    metric_keys = ['f1_macro', 'auprc', 'mcc', 'roc_auc', 'recall_fraud', 'f1_fraud']
    metric_labels = ['F1-Macro', 'AUPRC', 'MCC', 'ROC-AUC', 'Recall (Fraud)', 'F1 (Fraud)']
    
    spw_vals = [metrics_spw[k] for k in metric_keys]
    focal_vals = [metrics_focal[k] for k in metric_keys]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metric_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, spw_vals, width, label='scale_pos_weight',
                   color='#3498db', edgecolor='black', linewidth=0.7)
    bars2 = ax.bar(x + width/2, focal_vals, width, label='Focal Loss',
                   color='#e74c3c', edgecolor='black', linewidth=0.7)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Cost-Sensitive Strategies Comparison\nscale_pos_weight vs Custom Focal Loss',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xgb_strategy_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Strategy comparison plot saved")
