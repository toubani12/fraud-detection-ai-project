"""
Configuration file for the Fraud Detection Project
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Data
DATA_FILE = os.path.join(DATA_DIR, "creditcard.csv")
TARGET_COLUMN = "Class"
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"

# Reproducibility
RANDOM_STATE = 42

# Train/Test Split
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Imbalance ratio (approx)
FRAUD_CLASS = 1
NORMAL_CLASS = 0

# Model names
MODEL_LOGISTIC = "Logistic_Regression_ElasticNet"
MODEL_RF = "Random_Forest"
MODEL_XGBOOST = "XGBoost_CostSensitive"

# Optuna
N_TRIALS = 50
OPTUNA_TIMEOUT = 300  # seconds

# SMOTE/ADASYN
SAMPLING_STRATEGY = 0.1  # minority:majority ratio after resampling

# Thresholds
DECISION_THRESHOLD = 0.5

# Figure size defaults
FIG_SIZE = (12, 8)
FIG_DPI = 150
