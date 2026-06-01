# Credit Card Fraud Detection
## Projet Intelligence Artificielle — Classification Robuste en Environnement Déséquilibré

---

## Structure du Projet

```
fraud_detection_project/
│
├── data/
│   └── creditcard.csv              # Dataset (284 807 transactions)
│
├── src/
│   ├── config.py                   # Configuration centralisée (chemins, hyperparamètres)
│   ├── eda.py                      # Étape 1 : EDA, Feature Engineering, Resampling
│   ├── model_logistic.py           # Modèle 1 : Régression Logistique ElasticNet
│   ├── model_rf.py                 # Modèle 2 : Random Forest + Proximity Matrix
│   ├── model_xgboost.py            # Modèle 3 : XGBoost Cost-Sensitive + Optuna
│   ├── evaluation.py               # Métriques, calibration, visualisations
│   ├── interpretability.py         # Étape 4 : SHAP Analysis
│   └── main.py                     # Pipeline complète (orchestrateur)
│
├── outputs/                        # Modèles .pkl + toutes les visualisations
│   ├── model_logistic.pkl
│   ├── model_rf.pkl
│   ├── model_xgboost.pkl
│   ├── scaler.pkl
│   ├── eda_overview.png
│   ├── correlation_matrix.png
│   ├── vif_analysis.png
│   ├── resampling_comparison.png
│   ├── logistic_coefficients.png
│   ├── rf_feature_importance.png
│   ├── rf_proximity_outliers.png
│   ├── rf_outlier_explanation.txt
│   ├── optuna_convergence.png
│   ├── xgb_strategy_comparison.png
│   ├── xgb_feature_importance.png
│   ├── pr_roc_curves.png
│   ├── calibration_all_models.png
│   ├── calibration_platt_scaling.png
│   ├── shap_summary_beeswarm.png
│   ├── shap_bar_importance.png
│   ├── shap_dependence_top3.png
│   ├── shap_waterfall_fraud.png
│   ├── shap_class_comparison.png
│   ├── shap_logistic_importance.png
│   ├── model_comparison.png
│   └── results_summary.csv
│
├── reports/
│   └── rapport_projet_IA.md        # Rapport complet avec justifications
│
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dépendances principales :**
- `scikit-learn >= 1.3.0`
- `imbalanced-learn >= 0.11.0`
- `xgboost >= 2.0.0`
- `optuna >= 3.3.0`
- `shap >= 0.43.0`
- `pandas, numpy, matplotlib, seaborn`

---

## Lancement de la Pipeline Complète

```bash
python src/main.py
```

La pipeline exécute les 4 étapes :
1. EDA & Feature Engineering
2. Entraînement des 3 modèles
3. Évaluation & Calibration
4. Interprétabilité SHAP

---

## Résultats

| Modèle | F1-Macro | AUPRC | MCC | ROC-AUC |
|--------|----------|-------|-----|---------|
| Logistic (ElasticNet) | 0.4383 | 0.0520 | 0.0996 | 0.8426 |
| Random Forest | 0.4933 | 0.0375 | 0.1041 | 0.8399 |
| **XGBoost SPW** | **0.4958** | **0.0766** | **0.1033** | 0.8288 |

---

## Étapes du Projet

### Étape 1 : EDA & Préparation
- Analyse de la colinéarité (matrice de corrélation + VIF)
- Feature engineering avancé (14 nouvelles variables)
- Comparaison de 5 stratégies de rééquilibrage

### Étape 2 : Modèles
1. **Régression Logistique ElasticNet** — baseline linéaire avec L1+L2
2. **Random Forest + Matrice de Proximité** — analyse des outliers de prédiction
3. **XGBoost + Bayesian Search (Optuna)** — 2 stratégies cost-sensitive :
   - `scale_pos_weight` : pondération des gradients
   - `Focal Loss` : fonction de perte asymétrique personnalisée

### Étape 3 : Évaluation & Calibration
- Métriques : F1-Macro, AUPRC, MCC (pas d'Accuracy)
- Reliability Diagrams pour tous les modèles
- Platt Scaling sur le modèle mal calibré

### Étape 4 : Interprétabilité
- **TreeSHAP** pour XGBoost
- **Linear SHAP** pour la Régression Logistique
- Beeswarm, Waterfall, Dependence plots, Class comparison

---

## Utilisation du Modèle en Production

```python
import joblib
import numpy as np

# Charger le modèle et le scaler
model = joblib.load('outputs/model_xgboost.pkl')
scaler = joblib.load('outputs/scaler.pkl')

# Prédire sur nouvelles transactions
X_new_scaled = scaler.transform(X_new)
y_proba = model.predict_proba(X_new_scaled)[:, 1]

# Appliquer un seuil de décision
threshold = 0.5  # Ajuster selon le coût métier FP vs FN
y_pred = (y_proba >= threshold).astype(int)
```

---

*Projet IA — Credit Card Fraud Detection*  
*Technologies : Python, scikit-learn, XGBoost, Optuna, SHAP, imbalanced-learn*
