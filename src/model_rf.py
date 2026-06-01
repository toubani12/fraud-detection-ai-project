"""
Model 2: Random Forest with Proximity Matrix Analysis
- Train Random Forest for fraud detection
- Extract proximity matrix (co-leaf frequency)
- Detect prediction outliers from proximity
- Visualize and explain failure cases
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from evaluation import evaluate_model, print_metrics, plot_reliability_diagram

warnings.filterwarnings('ignore')


def compute_proximity_matrix(forest: RandomForestClassifier,
                              X: np.ndarray,
                              n_jobs: int = -1) -> np.ndarray:
    """
    Compute Random Forest proximity matrix.
    
    Proximity P[i,j] = fraction of trees where observations i and j
    end up in the same terminal leaf node.
    
    Interpretation:
    - High proximity: observations are "similar" according to the forest
    - Low proximity: observations are "dissimilar"
    
    Note: This implementation uses a memory-efficient batch approach.
    """
    n = X.shape[0]
    leaf_indices = forest.apply(X)  # shape: (n_samples, n_trees)
    
    # Vectorised co-leaf count
    proximity = np.zeros((n, n), dtype=np.float32)
    n_trees = leaf_indices.shape[1]
    
    # Batch to avoid memory overflow
    batch_size = min(500, n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # Compare each pair in batch vs all
        same_leaf = (leaf_indices[start:end, :, None] == 
                     leaf_indices[None, :, :].transpose(1, 2, 0))  
        # same_leaf shape: (batch, n_trees, n)
        proximity[start:end, :] = same_leaf.sum(axis=1) / n_trees
    
    return proximity


def detect_outliers_from_proximity(proximity: np.ndarray,
                                    y: np.ndarray,
                                    top_n: int = 50) -> np.ndarray:
    """
    Detect prediction outliers using proximity.
    
    Outlier score for class c: 
      outlier_score(i) = n / sum_j_in_c(proximity[i,j]^2)
    
    High outlier score → observation is far from its class neighbours
    → model likely struggles to classify it correctly
    """
    n = len(y)
    outlier_scores = np.zeros(n)
    
    for cls in [0, 1]:
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) == 0:
            continue
        # Sum of squared proximities within class
        prox_cls = proximity[np.ix_(cls_idx, cls_idx)]
        sum_sq = np.sum(prox_cls ** 2, axis=1)
        outlier_scores[cls_idx] = n / (sum_sq + 1e-10)
    
    return outlier_scores


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        output_dir: str) -> dict:
    """
    Train Random Forest with proximity analysis.
    
    Hyperparameter justification:
    - n_estimators=300: enough trees for stable proximity estimates 
      and low variance; diminishing returns beyond ~500
    - max_depth=20: deep enough to capture fraud patterns without 
      memorizing normal patterns (partial overfitting is OK with RF)
    - min_samples_leaf=5: prevents overly specific leaf nodes;
      helps compute meaningful proximity
    - class_weight='balanced': automatic compensation for 1:150 ratio
    - max_features='sqrt': standard RF heuristic, reduces correlation
      between trees
    """
    print("=" * 70)
    print("MODEL 2: RANDOM FOREST WITH PROXIMITY MATRIX")
    print("=" * 70)

    print("\n[INFO] Training Random Forest...")
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        oob_score=True
    )
    rf.fit(X_train, y_train)
    
    print(f"  OOB Score: {rf.oob_score_:.4f}")
    
    # Evaluate
    y_proba_test = rf.predict_proba(X_test)[:, 1]
    y_pred_test = rf.predict(X_test)
    metrics_test = evaluate_model(y_test, y_pred_test, y_proba_test, MODEL_RF)
    print_metrics(metrics_test)

    # Feature importance
    plot_feature_importance(rf, output_dir)

    # Calibration
    ece = plot_reliability_diagram(y_test, y_proba_test, 'Random Forest', output_dir)
    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")

    # ── Proximity Matrix & Outlier Analysis ──────────────────────────
    print("\n[INFO] Computing proximity matrix on test set (subsample)...")
    
    # Use a manageable subsample for proximity (full matrix is O(n^2))
    n_sample = min(2000, len(y_test))
    np.random.seed(RANDOM_STATE)
    sample_idx = np.concatenate([
        np.random.choice(np.where(y_test == 0)[0], 
                         min(n_sample - min(n_sample//10, np.sum(y_test==1)), 
                             np.sum(y_test==0)), replace=False),
        np.where(y_test == 1)[0]  # keep all fraud
    ])
    sample_idx = sample_idx[:n_sample]
    
    X_sample = X_test[sample_idx]
    y_sample = y_test[sample_idx]
    
    proximity = compute_proximity_matrix(rf, X_sample)
    print(f"  Proximity matrix computed: {proximity.shape}")
    
    # Outlier scores
    outlier_scores = detect_outliers_from_proximity(proximity, y_sample)
    
    # Analyze outliers
    analyze_prediction_outliers(rf, X_sample, y_sample, y_proba_test[sample_idx],
                                 outlier_scores, proximity, output_dir)

    return {
        'model': rf,
        'metrics': metrics_test,
        'y_proba': y_proba_test,
        'proximity': proximity,
        'outlier_scores': outlier_scores,
        'sample_idx': sample_idx,
        'ece': ece
    }


def analyze_prediction_outliers(rf, X: np.ndarray, y: np.ndarray,
                                  y_proba: np.ndarray, outlier_scores: np.ndarray,
                                  proximity: np.ndarray, output_dir: str,
                                  top_n: int = 20):
    """
    Visualize and explain prediction outliers.
    These are observations where the model hesitates or fails.
    """
    print(f"\n  Analyzing top {top_n} prediction outliers...")

    y_pred = (y_proba >= 0.5).astype(int)
    errors = (y_pred != y)
    
    # Outliers = high outlier score
    top_outlier_idx = np.argsort(outlier_scores)[-top_n:]
    
    # Categorize outlier types
    categories = []
    for idx in top_outlier_idx:
        is_error = errors[idx]
        true_label = y[idx]
        pred_label = y_pred[idx]
        proba = y_proba[idx]
        
        if is_error and true_label == 1:
            cat = 'False Negative (missed fraud)'
        elif is_error and true_label == 0:
            cat = 'False Positive (false alarm)'
        elif not is_error and proba > 0.3 and proba < 0.7:
            cat = 'Uncertain correct'
        else:
            cat = 'Structural outlier'
        categories.append(cat)

    print(f"\n  Outlier breakdown:")
    from collections import Counter
    for cat, cnt in Counter(categories).most_common():
        print(f"    {cat}: {cnt}")

    # ── Visualization ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. t-SNE of proximity matrix
    ax1 = fig.add_subplot(gs[0, :2])
    print("  Computing t-SNE projection of proximity...")
    # Dissimilarity = 1 - proximity
    dissimilarity = 1 - proximity
    np.fill_diagonal(dissimilarity, 0)
    dissimilarity = np.clip(dissimilarity, 0, 1)
    
    tsne = TSNE(n_components=2, metric='precomputed', random_state=RANDOM_STATE,
                perplexity=30, n_iter=500, init='random')
    embedding = tsne.fit_transform(dissimilarity)
    
    scatter_colors = np.where(y == 1, '#e74c3c', '#3498db')
    scatter_alpha = np.where(errors, 1.0, 0.4)
    scatter_size = np.where(errors, 80, 20)
    
    ax1.scatter(embedding[:, 0], embedding[:, 1],
                c=scatter_colors, alpha=0.5, s=20, label='')
    
    # Highlight outliers
    ax1.scatter(embedding[top_outlier_idx, 0], embedding[top_outlier_idx, 1],
                c='gold', s=150, marker='*', zorder=5, label='Prediction Outliers',
                edgecolors='black', linewidths=0.8)
    
    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Normal (True)'),
        Patch(facecolor='#e74c3c', label='Fraud (True)'),
        plt.scatter([], [], c='gold', s=100, marker='*', label='Top Outliers'),
    ]
    ax1.legend(handles=legend_elements[:2] + [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                   markersize=12, label='Top Outliers', markeredgecolor='black')
    ], fontsize=9)
    ax1.set_title('t-SNE of Proximity Matrix\n(Yellow stars = prediction outliers)',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')

    # 2. Outlier score distribution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(outlier_scores[y == 0], bins=40, alpha=0.6, color='#3498db',
             label='Normal', density=True)
    ax2.hist(outlier_scores[y == 1], bins=40, alpha=0.7, color='#e74c3c',
             label='Fraud', density=True)
    ax2.set_xlabel('Outlier Score', fontsize=11)
    ax2.set_title('Outlier Score Distribution\nby True Class', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Outlier score vs predicted probability
    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(y_proba, outlier_scores, c=y, cmap='RdBu_r',
                     alpha=0.5, s=15)
    ax3.scatter(y_proba[top_outlier_idx], outlier_scores[top_outlier_idx],
                c='gold', s=100, marker='*', zorder=5, edgecolors='black')
    ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Predicted Probability (Fraud)', fontsize=11)
    ax3.set_ylabel('Outlier Score', fontsize=11)
    ax3.set_title('Outlier Score vs Predicted Probability\n(Stars = Top Outliers)',
                  fontsize=12, fontweight='bold')
    plt.colorbar(sc, ax=ax3, label='True Class')
    ax3.grid(True, alpha=0.3)

    # 4. Error analysis
    ax4 = fig.add_subplot(gs[1, 1])
    error_types = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
    tn = np.sum((y == 0) & (y_pred < 0.5))
    fp = np.sum((y == 0) & (y_pred >= 0.5))
    fn = np.sum((y == 1) & (y_pred < 0.5))
    tp = np.sum((y == 1) & (y_pred >= 0.5))
    y_pred = (y_proba >= 0.5).astype(int)
    tn = np.sum((y == 0) & (y_pred == 0))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))
    tp = np.sum((y == 1) & (y_pred == 1))
    
    values = [tn, fp, fn, tp]
    colors_cm = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']
    bars = ax4.bar(error_types, values, color=colors_cm, edgecolor='black')
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:,}', ha='center', va='bottom', fontsize=9)
    ax4.set_xticklabels(error_types, rotation=15, ha='right', fontsize=9)
    ax4.set_title('Confusion Matrix Breakdown', fontsize=12, fontweight='bold')
    ax4.set_yscale('log')

    # 5. Proximity heatmap (fraud vs normal subsample)
    ax5 = fig.add_subplot(gs[1, 2])
    fraud_idx_local = np.where(y_sample == 1)[0][:30]
    normal_idx_local = np.random.choice(np.where(y_sample == 0)[0], 
                                         min(30, np.sum(y_sample==0)), replace=False)
    subset_idx = np.concatenate([fraud_idx_local, normal_idx_local])
    y_sample = y
    
    sub_prox = proximity[np.ix_(subset_idx, subset_idx)]
    labels = ['F' if y[i] == 1 else 'N' for i in subset_idx]
    
    im5 = ax5.imshow(sub_prox, cmap='hot', aspect='auto', vmin=0, vmax=1)
    n_fraud = len(fraud_idx_local)
    ax5.axhline(y=n_fraud - 0.5, color='cyan', linewidth=2)
    ax5.axvline(x=n_fraud - 0.5, color='cyan', linewidth=2)
    ax5.set_title('Proximity Heatmap\n(Fraud vs Normal subsample)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Observation')
    ax5.set_ylabel('Observation')
    plt.colorbar(im5, ax=ax5, label='Proximity')

    fig.suptitle('Random Forest — Proximity Matrix & Prediction Outlier Analysis',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(output_dir, 'rf_proximity_outliers.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Proximity outlier analysis saved")

    # Written explanation
    explanation = f"""
PREDICTION OUTLIER ANALYSIS — RANDOM FOREST
{'='*55}

Methodology:
  The proximity matrix P[i,j] counts the fraction of decision trees
  where observations i and j land in the same terminal leaf node.
  High proximity → observations are treated similarly by the model.

Outlier Score = n / Σ_j∈class(P[i,j]²)
  A high score means the observation is isolated from its class
  → the model lacks confident nearest neighbors → hesitation.

Top Outlier Categories Found:
{chr(10).join(f'  - {cat}: {cnt}' for cat, cnt in Counter(categories).most_common())}

Why the model fails on these points:
  1. False Negatives (missed fraud): These transactions share structural
     features with normal transactions (e.g., small amounts, common V-features)
     but have rare fraud patterns. The model lacks enough similar fraud
     examples nearby in feature space to vote confidently for "fraud".

  2. False Positives (false alarms): Normal transactions with unusual
     patterns (e.g., very high amounts, atypical hour) that resemble
     the fraud cluster — the model over-generalizes from limited fraud data.

  3. Uncertain correct: Observations near the decision boundary;
     the model's probability is close to 0.5 but the prediction is
     technically correct. These reveal the inherent ambiguity zone.

  4. Structural outliers: Observations that are inherently rare in their
     class — either anomalous fraud patterns or exceptional normal behavior.
     These push the model to explore unseen regions of feature space.
"""
    print(explanation)
    with open(os.path.join(output_dir, 'rf_outlier_explanation.txt'), 'w') as f:
        f.write(explanation)
    print(f"  ✓ Outlier explanation saved to rf_outlier_explanation.txt")


def plot_feature_importance(rf: RandomForestClassifier, output_dir: str, top_n: int = 25):
    """Plot MDI feature importances."""
    importances = rf.feature_importances_
    feature_names = [f'V{i}' for i in range(1, 29)]
    extra = ['Amount_log', 'Amount_sqrt', 'Hour', 'Hour_sin', 'Hour_cos',
             'Amount_bin_mean', 'Amount_bin_std', 'Amount_zscore_local',
             'V_norm', 'V_mean', 'V_std', 'V4_V11', 'V14_V17', 'V3_V10']
    feature_names = (feature_names + extra)[:len(importances)]

    idx = np.argsort(importances)[-top_n:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importances[idx], 
            color=plt.cm.RdYlGn(importances[idx] / importances[idx].max()),
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f'f{i}' 
                        for i in idx], fontsize=9)
    ax.set_xlabel('Mean Decrease in Impurity (MDI)', fontsize=12)
    ax.set_title(f'Random Forest — Top {top_n} Feature Importances',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rf_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Feature importance plot saved")
