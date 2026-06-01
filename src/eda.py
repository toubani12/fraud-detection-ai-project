"""
Step 1: Exploratory Data Analysis (EDA) and Data Preparation
- Correlation matrix & VIF analysis
- Class imbalance visualization
- Feature engineering
- Imbalance treatment comparison: class_weight vs SMOTE/ADASYN
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
import warnings
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

warnings.filterwarnings('ignore')


def load_and_inspect(filepath: str) -> pd.DataFrame:
    """Load dataset and print basic statistics."""
    print("=" * 70)
    print("STEP 1: EXPLORATORY DATA ANALYSIS & DATA PREPARATION")
    print("=" * 70)

    df = pd.read_csv(filepath)
    print(f"\n[INFO] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n{'─'*40}")
    print("BASIC STATISTICS")
    print(f"{'─'*40}")
    print(df.describe().round(3).to_string())
    
    print(f"\n{'─'*40}")
    print("MISSING VALUES")
    print(f"{'─'*40}")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values detected ✓")
    else:
        print(missing[missing > 0])

    print(f"\n{'─'*40}")
    print("CLASS DISTRIBUTION")
    print(f"{'─'*40}")
    counts = df[TARGET_COLUMN].value_counts()
    print(f"  Normal (0): {counts[0]:>10,}  ({counts[0]/len(df)*100:.4f}%)")
    print(f"  Fraud  (1): {counts[1]:>10,}  ({counts[1]/len(df)*100:.4f}%)")
    print(f"  Imbalance Ratio: {counts[0]/counts[1]:.1f}:1")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced feature engineering:
    - Log-transform Amount (skewed)
    - Time cyclic features (hour of day)
    - Interaction features
    - Statistical aggregations
    """
    print(f"\n{'─'*40}")
    print("FEATURE ENGINEERING")
    print(f"{'─'*40}")

    df = df.copy()

    # 1. Log-transform Amount (heavy right skew)
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Amount_sqrt'] = np.sqrt(df['Amount'])
    print("  ✓ Amount transformations: log1p, sqrt")

    # 2. Time features
    df['Hour'] = (df['Time'] % 86400) / 3600  # hour of day
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    print("  ✓ Cyclic time features: sin/cos of hour")

    # 3. Rolling statistics on Amount (by time bins)
    time_bins = pd.cut(df['Time'], bins=50, labels=False)
    df['Amount_bin_mean'] = df.groupby(time_bins)['Amount'].transform('mean')
    df['Amount_bin_std'] = df.groupby(time_bins)['Amount'].transform('std').fillna(0)
    df['Amount_zscore_local'] = (
        (df['Amount'] - df['Amount_bin_mean']) / 
        (df['Amount_bin_std'] + 1e-8)
    )
    print("  ✓ Local z-score of Amount within time bins")

    # 4. L2-norm of PCA components (V1-V28)
    v_cols = [f'V{i}' for i in range(1, 29)]
    df['V_norm'] = np.linalg.norm(df[v_cols].values, axis=1)
    df['V_mean'] = df[v_cols].mean(axis=1)
    df['V_std'] = df[v_cols].std(axis=1)
    print("  ✓ PCA vector statistics: L2-norm, mean, std")

    # 5. Top interactions (V4*V11 known fraud indicators)
    df['V4_V11'] = df['V4'] * df['V11']
    df['V14_V17'] = df['V14'] * df['V17']
    df['V3_V10'] = df['V3'] * df['V10']
    print("  ✓ Multiplicative interaction features")

    # 6. Drop raw Time and Amount (replaced by engineered versions)
    df = df.drop(columns=['Time', 'Amount'])
    print(f"\n  Total features after engineering: {df.shape[1]-1} (excl. target)")

    return df


def analyze_correlations(df: pd.DataFrame, output_dir: str):
    """Compute and visualize correlation matrix + VIF."""
    print(f"\n{'─'*40}")
    print("CORRELATION & MULTICOLLINEARITY ANALYSIS")
    print(f"{'─'*40}")

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols]

    # ── Correlation matrix ──────────────────────────────────────────
    corr = X.corr()
    # Find high correlations
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [(upper.index[r], upper.columns[c], upper.iloc[r, c])
                 for r, c in zip(*np.where(np.abs(upper) > 0.7))]
    if high_corr:
        print(f"\n  High correlations (|r| > 0.7):")
        for f1, f2, v in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:10]:
            print(f"    {f1:20s} ↔ {f2:20s} : {v:.3f}")
    else:
        print("  No high correlations (|r| > 0.7) found ✓")

    # Plot correlation heatmap (top 20 features by variance)
    top_features = X.var().nlargest(20).index.tolist()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr.loc[top_features, top_features], dtype=bool))
    sns.heatmap(
        corr.loc[top_features, top_features],
        mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 7},
        vmin=-1, vmax=1
    )
    ax.set_title('Feature Correlation Matrix (Top 20 by Variance)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Correlation heatmap saved")

    # ── VIF Analysis ─────────────────────────────────────────────────
    print("\n  Computing VIF (Variance Inflation Factor)...")
    # Use subset to avoid memory issues
    vif_features = [f'V{i}' for i in range(1, 16)] + ['Amount_log', 'V_norm']
    vif_data = X[vif_features].copy()
    vif_data = (vif_data - vif_data.mean()) / (vif_data.std() + 1e-8)

    vif_df = pd.DataFrame({
        'Feature': vif_features,
        'VIF': [variance_inflation_factor(vif_data.values, i) 
                for i in range(len(vif_features))]
    }).sort_values('VIF', ascending=False)

    print("\n  VIF Results (threshold = 10):")
    print(vif_df.to_string(index=False))

    high_vif = vif_df[vif_df['VIF'] > 10]
    if len(high_vif) > 0:
        print(f"\n  ⚠️  {len(high_vif)} features with VIF > 10 (potential multicollinearity)")
    else:
        print("\n  All VIF values < 10 — no severe multicollinearity ✓")

    # Plot VIF
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if v > 10 else '#2ecc71' if v < 5 else '#f39c12' 
              for v in vif_df['VIF']]
    bars = ax.barh(vif_df['Feature'], vif_df['VIF'], color=colors)
    ax.axvline(x=5, color='orange', linestyle='--', linewidth=1.5, label='VIF=5 (moderate)')
    ax.axvline(x=10, color='red', linestyle='--', linewidth=1.5, label='VIF=10 (high)')
    ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12)
    ax.set_title('VIF Analysis — Multicollinearity Detection', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vif_analysis.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ VIF chart saved")

    return vif_df


def plot_class_imbalance(df: pd.DataFrame, output_dir: str):
    """Visualize class imbalance and feature distributions by class."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Class counts
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df[TARGET_COLUMN].value_counts()
    bars = ax1.bar(['Normal (0)', 'Fraud (1)'], counts.values,
                   color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{val:,}\n({val/counts.sum()*100:.3f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_title('Class Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_yscale('log')

    # 2. Pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pie(counts.values, labels=['Normal', 'Fraud'],
            colors=['#3498db', '#e74c3c'], autopct='%1.3f%%',
            startangle=90, explode=(0, 0.1))
    ax2.set_title('Imbalance Ratio\n(log scale)', fontsize=12, fontweight='bold')

    # 3. Amount distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if 'Amount_log' in df.columns:
        col = 'Amount_log'
    else:
        col = 'Amount'
    df[df[TARGET_COLUMN] == 0][col].hist(bins=50, alpha=0.6, color='#3498db',
                                          label='Normal', density=True, ax=ax3)
    df[df[TARGET_COLUMN] == 1][col].hist(bins=50, alpha=0.7, color='#e74c3c',
                                          label='Fraud', density=True, ax=ax3)
    ax3.set_title(f'{col} Distribution by Class', fontsize=12, fontweight='bold')
    ax3.set_xlabel(col)
    ax3.legend()

    # 4. V14 distribution (one of most discriminative)
    ax4 = fig.add_subplot(gs[1, 0])
    for cls, color, label in [(0, '#3498db', 'Normal'), (1, '#e74c3c', 'Fraud')]:
        data = df[df[TARGET_COLUMN] == cls]['V14'].clip(-10, 10)
        ax4.hist(data, bins=60, alpha=0.65, color=color, label=label, density=True)
    ax4.set_title('V14 Distribution by Class', fontsize=12, fontweight='bold')
    ax4.legend()

    # 5. V4 distribution
    ax5 = fig.add_subplot(gs[1, 1])
    for cls, color, label in [(0, '#3498db', 'Normal'), (1, '#e74c3c', 'Fraud')]:
        data = df[df[TARGET_COLUMN] == cls]['V4'].clip(-10, 10)
        ax5.hist(data, bins=60, alpha=0.65, color=color, label=label, density=True)
    ax5.set_title('V4 Distribution by Class', fontsize=12, fontweight='bold')
    ax5.legend()

    # 6. Box plot of V_norm
    ax6 = fig.add_subplot(gs[1, 2])
    if 'V_norm' in df.columns:
        data_0 = df[df[TARGET_COLUMN] == 0]['V_norm'].clip(0, 30)
        data_1 = df[df[TARGET_COLUMN] == 1]['V_norm'].clip(0, 30)
        ax6.boxplot([data_0, data_1], labels=['Normal', 'Fraud'],
                    patch_artist=True,
                    boxprops=dict(facecolor='#3498db'),
                    medianprops=dict(color='white', linewidth=2))
        ax6.set_title('V_norm (L2) Distribution', fontsize=12, fontweight='bold')
        ax6.set_ylabel('V_norm')

    fig.suptitle('Exploratory Data Analysis — Credit Card Fraud Detection',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(output_dir, 'eda_overview.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ EDA overview plot saved")


def compare_resampling_strategies(X_train: np.ndarray, y_train: np.ndarray,
                                   output_dir: str) -> dict:
    """
    Compare imbalance treatment strategies:
    1. No resampling (raw)
    2. Class weights (algorithmic)
    3. SMOTE (oversampling)
    4. ADASYN (adaptive oversampling)
    5. NearMiss (undersampling)
    """
    print(f"\n{'─'*40}")
    print("IMBALANCE TREATMENT STRATEGIES COMPARISON")
    print(f"{'─'*40}")

    results = {}

    # Strategy 1: Raw data
    results['raw'] = (X_train, y_train, None)
    print(f"  Raw:        {np.sum(y_train==0):>7,} normal | {np.sum(y_train==1):>5,} fraud")

    # Strategy 2: Class weight — no resampling needed (passed to model)
    results['class_weight'] = (X_train, y_train, 'balanced')
    print(f"  ClassWeight: algorithmic — no resampling required")

    # Strategy 3: SMOTE
    print("  Applying SMOTE...", end=' ')
    smote = SMOTE(sampling_strategy=SAMPLING_STRATEGY, random_state=RANDOM_STATE)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    results['smote'] = (X_smote, y_smote, None)
    print(f"Done → {np.sum(y_smote==0):>7,} normal | {np.sum(y_smote==1):>5,} fraud")

    # Strategy 4: ADASYN
    print("  Applying ADASYN...", end=' ')
    try:
        adasyn = ADASYN(sampling_strategy=SAMPLING_STRATEGY, random_state=RANDOM_STATE)
        X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
        results['adasyn'] = (X_adasyn, y_adasyn, None)
        print(f"Done → {np.sum(y_adasyn==0):>7,} normal | {np.sum(y_adasyn==1):>5,} fraud")
    except Exception as e:
        print(f"Warning: {e} — using SMOTE as fallback")
        results['adasyn'] = results['smote']

    # Strategy 5: NearMiss (undersampling)
    print("  Applying NearMiss...", end=' ')
    nm = NearMiss(version=1, sampling_strategy=SAMPLING_STRATEGY)
    X_nm, y_nm = nm.fit_resample(X_train, y_train)
    results['nearmiss'] = (X_nm, y_nm, None)
    print(f"Done → {np.sum(y_nm==0):>7,} normal | {np.sum(y_nm==1):>5,} fraud")

    # Visualization
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    strategies = {
        'Raw Data': y_train,
        'Class Weight\n(Algorithmic)': y_train,
        'SMOTE': y_smote,
        'ADASYN': y_adasyn if 'adasyn' in results else y_smote,
        'NearMiss': y_nm
    }
    colors_map = {0: '#3498db', 1: '#e74c3c'}
    for ax, (name, y) in zip(axes, strategies.items()):
        counts = np.bincount(y)
        bars = ax.bar(['Normal', 'Fraud'],
                      [counts[0], counts[1]],
                      color=['#3498db', '#e74c3c'],
                      edgecolor='black', linewidth=0.7)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_yscale('log')
        for bar, val in zip(bars, [counts[0], counts[1]]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    f'{val:,}', ha='center', va='bottom', fontsize=8)
        ratio = counts[0] / counts[1]
        ax.set_xlabel(f'Ratio: {ratio:.1f}:1', fontsize=8)

    fig.suptitle('Imbalance Treatment Strategies Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resampling_comparison.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Resampling comparison plot saved")

    return results


def prepare_data(df: pd.DataFrame):
    """Split data into train/validation/test sets with stratification."""
    print(f"\n{'─'*40}")
    print("DATA SPLITTING")
    print(f"{'─'*40}")

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values

    # Stratified split: 70% train, 10% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"  Train: {X_train.shape[0]:>7,} samples | Fraud: {y_train.sum():>5,} ({y_train.mean():.4%})")
    print(f"  Val:   {X_val.shape[0]:>7,} samples | Fraud: {y_val.sum():>5,} ({y_val.mean():.4%})")
    print(f"  Test:  {X_test.shape[0]:>7,} samples | Fraud: {y_test.sum():>5,} ({y_test.mean():.4%})")

    # Scale features using RobustScaler (robust to outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print(f"  ✓ Features scaled with RobustScaler")

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            feature_cols, scaler)


def run_eda(output_dir: str):
    """Run complete EDA pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # Load
    df = load_and_inspect(DATA_FILE)

    # Feature engineering
    df = feature_engineering(df)

    # Plots
    plot_class_imbalance(df, output_dir)

    # Correlation
    vif_df = analyze_correlations(df, output_dir)

    # Split
    (X_train_s, X_val_s, X_test_s,
     X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_cols, scaler) = prepare_data(df)

    # Resampling strategies
    resampling = compare_resampling_strategies(X_train_s, y_train, output_dir)

    print(f"\n{'='*70}")
    print("EDA COMPLETE ✓")
    print(f"{'='*70}")

    return {
        'df': df,
        'X_train': X_train_s,
        'X_val': X_val_s,
        'X_test': X_test_s,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'resampling': resampling,
        'vif': vif_df
    }


if __name__ == '__main__':
    data = run_eda(OUTPUT_DIR)
