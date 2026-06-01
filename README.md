# Classification Robuste et Analyse de Décision en Environnement Critique
## Credit Card Fraud Detection — Projet Intelligence Artificielle

**Auteur :** TOUBANI Badr Eddine | **Filière :** SDIA | **Établissement :** ENSET Mohammedia  
**Encadrant :** Mme. Asmae OUHMIDA | **Année :** 2025-2026

---

## Abstract

Ce projet présente une pipeline complète de machine learning pour la détection de fraude bancaire sur le dataset *Credit Card Fraud Detection* (Kaggle, 284 807 transactions, ratio 150:1). Trois approches mathématiquement distinctes sont confrontées — Régression Logistique ElasticNet, Random Forest avec analyse de proximité, et XGBoost avec apprentissage sensible au coût — en s'appuyant sur 14 features engineerées, une optimisation bayésienne (Optuna/TPE), une calibration par Platt Scaling, et une interprétabilité par SHAP. Le meilleur modèle (XGBoost scale_pos_weight, AUPRC = 0.0766, F1-Macro = 0.4958, MCC = 0.1033) surpasse significativement la baseline aléatoire (AUPRC ≈ 0.0066, soit ×11.6) tout en produisant des probabilités exploitables pour la décision à seuil variable.

**Mots-clés :** déséquilibre de classes, détection de fraude, Elastic Net, Random Forest, XGBoost, Focal Loss, Optuna, SHAP, Platt Scaling, calibration

---

## Structure du Projet

```
fraud_detection_project/
├── data/
│   └── creditcard.csv              # 284 807 transactions (Kaggle)
├── src/
│   ├── config.py                   # Chemins, hyperparamètres, constantes
│   ├── eda.py                      # Étape 1 : EDA, Feature Engineering, Resampling
│   ├── model_logistic.py           # Modèle 1 : Régression Logistique ElasticNet
│   ├── model_rf.py                 # Modèle 2 : Random Forest + Proximity Matrix
│   ├── model_xgboost.py            # Modèle 3 : XGBoost Cost-Sensitive + Optuna
│   ├── evaluation.py               # Métriques F1/AUPRC/MCC, calibration, visualisations
│   ├── interpretability.py         # Étape 4 : TreeSHAP + LinearSHAP
│   └── main.py                     # Orchestrateur de la pipeline complète
├── notebooks/
│   ├── 01_EDA.ipynb                # EDA interactive et feature engineering
│   ├── 02_Models.ipynb             # Entraînement et analyse des 3 modèles
│   ├── 03_Evaluation_Calibration.ipynb  # Métriques, PR/ROC, reliability diagrams
│   └── 04_Interpretability.ipynb   # SHAP beeswarm, waterfall, dependence plots
├── outputs/                        # Modèles .pkl + 20+ visualisations générées
├── reports/
│   └── rapport_projet_IA.md        # Rapport théorique complet (25 pages équivalent)
├── pyproject.toml                  # Dépendances uv (remplace requirements.txt)
└── uv.lock                         # Lockfile reproductible
```

---

## Installation (uv)

> Ce projet utilise **[uv](https://github.com/astral-sh/uv)** pour la gestion des dépendances.

```bash
# 1. Installer uv (macOS/Linux)
brew install uv

# 2. Installer libomp pour XGBoost (macOS uniquement)
brew install libomp

# 3. Créer l'environnement et installer les dépendances
uv sync

# 4. Lancer la pipeline complète
uv run python src/main.py

# 5. Ouvrir les notebooks
uv run jupyter notebook notebooks/
```

**Dépendances principales :** `scikit-learn ≥ 1.3`, `imbalanced-learn ≥ 0.11`, `xgboost ≥ 2.0`, `optuna ≥ 3.3`, `shap ≥ 0.43`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `jupyter`

---

## Résultats

### Tableau comparatif final

| Modèle | F1-Macro | F1-Fraude | AUPRC | ROC-AUC | MCC | Recall-Fraude |
|--------|----------|-----------|-------|---------|-----|---------------|
| Logistic (ElasticNet) | 0.4383 | 0.0381 | 0.0520 | 0.8426 | 0.0996 | 82.98% |
| Random Forest | 0.4933 | 0.0545 | 0.0375 | 0.8399 | 0.1041 | 55.32% |
| **XGBoost SPW ★** | **0.4958** | 0.0533 | **0.0766** | 0.8231 | 0.1033 | 53.19% |
| XGBoost Focal | 0.4958 | 0.0506 | 0.0695 | 0.8231 | 0.0970 | 50.00% |

> ★ Meilleur modèle par AUPRC (métrique principale pour données déséquilibrées).  
> Baseline aléatoire : AUPRC ≈ 0.0066 — XGBoost SPW le dépasse de ×11.6.

### Calibration (Expected Calibration Error)

| Modèle | ECE Avant | Action | ECE Après |
|--------|-----------|--------|-----------|
| Logistic (ElasticNet) | 0.4756 | Platt Scaling | **0.0602** |
| Random Forest | 0.3859 | — | 0.3859 |
| XGBoost SPW | 0.4631 | — | 0.4631 |

### SPW vs Focal Loss (sur validation Optuna)

| Stratégie | AUPRC val | Implémentation | Stabilité |
|-----------|-----------|----------------|-----------|
| scale_pos_weight | **0.1733** | Native XGBoost | Stable |
| Focal Loss | 0.0043 | Custom gradient | Instable (gradient approx.) |

---

## 1. Problématique

La détection de fraude bancaire est un problème de classification binaire avec un **déséquilibre extrême** : 1 882 fraudes sur 284 807 transactions (0.66%), ratio 150.3:1. Un modèle naïf prédisant toujours "Normal" atteint 99.34% d'accuracy sans détecter une seule fraude — l'accuracy est donc une métrique trompeuse. Ce projet adresse ce problème par trois angles complémentaires : gestion du déséquilibre au niveau données et algorithmique, métriques adaptées (F1-Macro, AUPRC, MCC), et calibration des probabilités pour une prise de décision à seuil variable.

---

## 2. Dataset

Le dataset *Credit Card Fraud Detection* (Kaggle / ULB Machine Learning Group) contient :

| Attribut | Valeur |
|----------|--------|
| Transactions | 284 807 (2 jours) |
| Features initiales | 30 (V1–V28 PCA + Time + Amount) |
| Cible | Class (0 = Normal, 1 = Fraude) |
| Classe majoritaire | 282 925 (99.34%) |
| Classe minoritaire | 1 882 (0.66%) |
| Ratio | 150.3:1 |
| Valeurs manquantes | 0 |

---

## 3. Méthodes

### 3.1 Feature Engineering (42 features au total)

| Feature créée | Formule | Justification |
|---------------|---------|---------------|
| `Amount_log` | log(1 + Amount) | Normalise la distribution fortement asymétrique |
| `Amount_sqrt` | √Amount | Transformation douce, préserve les petits montants |
| `Hour_sin/cos` | sin/cos(2π·h/24) | Encodage cyclique, évite la discontinuité 23h→0h |
| `Amount_zscore_local` | (x − μ_bin)/σ_bin | Anomalie relative dans la fenêtre temporelle (50 bins) |
| `V_norm` | ‖V1,...,V28‖₂ | Amplitude globale du profil PCA |
| `V_mean`, `V_std` | — | Statistiques de tendance centrale des composantes PCA |
| `V4_V11`, `V14_V17`, `V3_V10` | Vᵢ × Vⱼ | Interactions non-linéaires entre composantes discriminantes |

### 3.2 Colinéarité détectée

| Feature 1 | Feature 2 | Corrélation |
|-----------|-----------|-------------|
| V2 | V13 | 1.000 (colinéarité parfaite, VIF = ∞) |
| V24 | V26 | 1.000 (colinéarité parfaite, VIF = ∞) |
| V_norm | V_std | 0.990 |
| Amount_log | Amount_sqrt | 0.811 |
| Hour | Hour_sin | −0.780 |

**Décision :** Pas de suppression manuelle. La régularisation ElasticNet (L2) gère la multicolinéarité pour LR ; le sous-échantillonnage aléatoire des features le fait pour RF et XGBoost.

### 3.3 Stratégies de rééquilibrage comparées

| Stratégie | Normal | Fraude | Ratio final |
|-----------|--------|--------|-------------|
| Raw | 198 046 | 1 318 | 150:1 |
| class_weight (algorithmique) | 198 046 | 1 318 | 150:1 |
| SMOTE | 198 046 | 19 804 | 10:1 |
| ADASYN | 198 046 | 20 359 | ~10:1 |
| NearMiss | 13 180 | 1 318 | 10:1 |

**Retenu :** `class_weight='balanced'` / `scale_pos_weight` — agit sur l'optimisation sans modifier la distribution des données.

### 3.4 Modèle 1 — Régression Logistique ElasticNet

**Formulation :** `min_w Σ log(1 + exp(−yᵢ wᵀxᵢ)) + α[ρ‖w‖₁ + (1−ρ)/2 ‖w‖₂²]`

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| C | 0.1 | Régularisation modérée, sélectionné par CV 5-fold AUPRC |
| l1_ratio | 0.5 | Équilibre L1 (sparsité) / L2 (stabilité multicolinéarité) |
| solver | saga | Seul solver compatible ElasticNet pour grands datasets |
| class_weight | balanced | w_k = n_samples / (n_classes × n_k) |
| max_iter | 2000 | Garantit la convergence de SAGA au ratio 150:1 |

### 3.5 Modèle 2 — Random Forest avec Analyse de Proximité

**Matrice de proximité :** `P[i,j] = (1/T) Σ 𝟙[leaf_t(i) = leaf_t(j)]`

**Score d'outlier :** `OutlierScore(i) = n / Σ_{j∈classe} P[i,j]²`

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| n_estimators | 100 | Compromis stabilité/vitesse |
| max_depth | 12 | Capture les patterns de fraude complexes |
| min_samples_leaf | 5 | Feuilles significatives pour proximité valide |
| max_features | sqrt(p) | ≈6 features/nœud, réduit la corrélation inter-arbres |
| class_weight | balanced | Compensation automatique 150:1 |

### 3.6 Modèle 3 — XGBoost Cost-Sensitive + Optuna

**Stratégie A — scale_pos_weight :** `SPW = N_normal / N_fraud ≈ 150` → optimal découvert = **157.7**

**Stratégie B — Focal Loss :** `FL(pₜ) = −α(1−pₜ)^γ log(pₜ)`, γ=2 (Lin et al., 2017)

**Optimisation Optuna (TPE, 30 trials) :**

| Hyperparamètre | Plage | Optimal |
|----------------|-------|---------|
| max_depth | [3, 8] | 4 |
| learning_rate | [0.05, 0.3] log | 0.128 |
| n_estimators | [100, 400] | 250 |
| subsample | [0.6, 1.0] | 0.716 |
| colsample_bytree | [0.6, 1.0] | 0.845 |
| reg_lambda | [1, 8] | 1.98 |
| min_child_weight | [1, 8] | 3 |
| scale_pos_weight | [75, 300] | 157.7 |

### 3.7 Métriques d'évaluation

| Métrique | Formule | Justification |
|----------|---------|---------------|
| **F1-Macro** | ½(F1₀ + F1₁) | Équilibre précision/rappel, insensible au déséquilibre |
| **AUPRC** | ∫P(r)dr | Mesure directe de détection sans fausses alarmes ; baseline = 0.0066 |
| **MCC** | (TP·TN−FP·FN)/√(...) | Résumé non-biaisé de la matrice de confusion complète |

**Accuracy exclue** : un modèle trivial atteint 99.34% sans jamais détecter de fraude.

### 3.8 Calibration & Platt Scaling

`P_cal(y=1 | f(x)) = 1/(1 + exp(−(A·f(x) + B)))` — paramètres A, B estimés par max de vraisemblance (CV 2 folds).

Résultat sur Logistic Regression : ECE **0.4756 → 0.0602** (réduction de 87%).

### 3.9 Interprétabilité SHAP

`φⱼ(i) = Σ_{S⊆F\{j}} [|S|!(|F|−|S|−1)!/|F|!] × [v(S∪{j})−v(S)]`

- **TreeSHAP** (XGBoost) : exact, O(TLD²) — évite l'approximation KernelSHAP
- **LinearSHAP** (Régression Logistique) : exploite la structure analytique du modèle

**Features les plus discriminantes (mean |SHAP|) :** Amount_log, Amount_sqrt, Amount_zscore_local, V28, V6, V_norm, V14, V4, V17

---

## 4. Visualisations Générées

| Fichier | Description |
|---------|-------------|
| `eda_overview.png` | Distribution des classes, Amount_log, V14, V4, V_norm |
| `correlation_matrix.png` | Heatmap de corrélation (top 20 features par variance) |
| `vif_analysis.png` | VIF par feature avec seuils 5 et 10 |
| `resampling_comparison.png` | Comparaison des 5 stratégies de rééquilibrage |
| `logistic_coefficients.png` | Top 25 coefficients ElasticNet (rouge=fraude, bleu=normal) |
| `rf_feature_importance.png` | MDI — Random Forest (top 25) |
| `rf_proximity_outliers.png` | t-SNE, heatmap de proximité, outlier scores |
| `optuna_convergence.png` | Historique de convergence SPW et Focal Loss |
| `xgb_strategy_comparison.png` | SPW vs Focal Loss sur toutes les métriques |
| `xgb_feature_importance.png` | Gain normalisé XGBoost (top 25) |
| `pr_roc_curves.png` | Courbes PR et ROC pour les 4 modèles |
| `calibration_*.png` | Reliability Diagrams par modèle |
| `calibration_platt_scaling.png` | Avant/après Platt Scaling (LR) |
| `shap_summary_beeswarm.png` | Beeswarm plot — direction et magnitude des SHAP values |
| `shap_bar_importance.png` | Importance globale (mean \|SHAP\|) top 20 |
| `shap_dependence_top3.png` | Dependence plots pour les 3 top features |
| `shap_waterfall_fraud.png` | Explication d'une prédiction frauduleuse individuelle |
| `shap_class_comparison.png` | SHAP moyen Fraude vs Normal (top 20 features) |
| `shap_logistic_importance.png` | SHAP Logistic Regression (LinearExplainer) |
| `model_comparison.png` | Comparaison visuelle F1-Macro, AUPRC, MCC, ROC-AUC |
| `results_summary.csv` | Tableau de métriques complet, toutes variantes |

---

## 5. Utilisation en Production

```python
import joblib
import numpy as np

# Charger le modèle XGBoost SPW (meilleur AUPRC) et le scaler
model  = joblib.load('outputs/model_xgboost.pkl')
scaler = joblib.load('outputs/scaler.pkl')

# Pré-traiter et prédire
X_new_scaled = scaler.transform(X_new_engineered)
y_proba = model.predict_proba(X_new_scaled)[:, 1]

# Seuil calibré selon le coût métier C(FN) >> C(FP)
# Recommandation : optimiser threshold = argmin[C(FP)×FP + C(FN)×FN]
threshold = 0.3  # Plus bas que 0.5 pour maximiser le recall fraude
y_pred = (y_proba >= threshold).astype(int)
```

**Recommandations opérationnelles :**

| Aspect | Recommandation |
|--------|----------------|
| Modèle principal | XGBoost SPW + Platt Scaling (probabilités calibrées) |
| Seuil | Calibrer selon ratio coût FN/FP (coût fraude vs blocage injustifié) |
| Explicabilité | SHAP pour justifier chaque décision de blocage |
| Monitoring | Surveiller AUPRC et MCC en production ; réentraîner si dégradation > 5% |
| Drift | KS test / PSI pour détecter les changements de distribution |
| Réentraînement | Mensuel avec les nouvelles transactions labellisées |

---

## 6. Références

1. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016.
3. Lin, T. Y., et al. (2017). *Focal Loss for Dense Object Detection*. ICCV 2017.
4. Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-Sampling Technique*. JAIR, 16.
5. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
6. Platt, J. C. (1999). *Probabilistic Outputs for Support Vector Machines*. Adv. Large Margin Classifiers.
7. Akiba, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD 2019.
8. Lundberg, S. M., et al. (2020). *From Local Explanations to Global Understanding with Explainable AI for Trees*. Nature Machine Intelligence, 2(1), 56–67.
9. Dal Pozzolo, A., et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI 2015.
10. He, H., & Garcia, E. A. (2009). *Learning from Imbalanced Data*. IEEE TKDE, 21(9).

---

*Projet IA — ENSET Mohammedia | SDIA 2025-2026 | Mme. Asmae OUHMIDA*  
*Technologies : Python 3.14 · uv · scikit-learn · XGBoost · Optuna · SHAP · imbalanced-learn*
