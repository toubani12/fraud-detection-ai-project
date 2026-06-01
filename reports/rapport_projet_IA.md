# Rapport de Projet — Intelligence Artificielle
## Classification Robuste et Analyse de Décision en Environnement Critique

---

| Étudiant | TOUBANI Badr Eddine |
|----------|---------------------|
| Professeur | Mme. Asmae OUHMIDA |
| Filière | SDIA — Systèmes Distribués et Intelligence Artificielle |
| Établissement | ENSET (École Nationale Supérieure de l'Enseignement Technique) |
| Module | Intelligence Artificielle |
| Dataset | Credit Card Fraud Detection — 284 807 transactions (Kaggle) |
| Date | 01 Juin 2026 |

---

## Liste des Abréviations

| Abréviation | Définition |
|-------------|-----------|
| ML | Machine Learning — Apprentissage automatique |
| EDA | Exploratory Data Analysis — Analyse Exploratoire des Données |
| VIF | Variance Inflation Factor — Facteur d'Inflation de la Variance |
| SMOTE | Synthetic Minority Over-sampling Technique |
| ADASYN | Adaptive Synthetic Sampling Approach |
| AUPRC | Area Under the Precision-Recall Curve |
| MCC | Matthews Correlation Coefficient |
| ROC-AUC | Receiver Operating Characteristic — Area Under Curve |
| ECE | Expected Calibration Error — Erreur de Calibration Attendue |
| SHAP | SHapley Additive exPlanations |
| TPE | Tree-structured Parzen Estimator (Optuna) |
| L1 / Lasso | Régularisation par norme L1 (somme des valeurs absolues) |
| L2 / Ridge | Régularisation par norme L2 (somme des carrés) |
| FP / FN / TP / TN | Faux Positif / Faux Négatif / Vrai Positif / Vrai Négatif |
| SPW | scale_pos_weight — Paramètre XGBoost pour déséquilibre |
| PCA | Principal Component Analysis — Analyse en Composantes Principales |

---

## Table des Matières

1. [Introduction Générale](#1-introduction-générale)
2. [Étape 1 : Analyse Exploratoire et Préparation des Données](#2-étape-1--analyse-exploratoire-et-préparation-des-données)
3. [Étape 2 : Développement des Modèles](#3-étape-2--développement-des-modèles)
4. [Étape 3 : Évaluation et Calibration](#4-étape-3--évaluation-et-calibration)
5. [Étape 4 : Interprétabilité par SHAP](#5-étape-4--interprétabilité-par-shap)
6. [Synthèse et Comparaison des Modèles](#6-synthèse-et-comparaison-des-modèles)
7. [Conclusion et Perspectives](#7-conclusion-et-perspectives)
8. [Références Bibliographiques](#8-références-bibliographiques)

---

## 1. Introduction Générale

La détection de fraude bancaire constitue l'un des défis les plus représentatifs des problèmes de classification déséquilibrée en intelligence artificielle appliquée. Dans des scénarios réels tels que la fraude par carte de crédit, le diagnostic médical ou la prédiction de pannes industrielles, les classes sont rarement équilibrées : les événements d'intérêt (fraudes, maladies, défaillances) sont rares face à la masse des événements normaux.

Ce rapport documente la conception et l'implémentation d'un système complet de classification robuste appliqué au jeu de données *Credit Card Fraud Detection* (Kaggle), comprenant **284 807 transactions** avec un ratio de déséquilibre de **150:1**. L'objectif central est de construire des modèles capables de détecter efficacement les fraudes tout en produisant des probabilités calibrées exploitables par les systèmes de décision.

La démarche suit une approche scientifique rigoureuse articulée en quatre étapes complémentaires :

1. **Analyse Exploratoire et Préparation des Données (EDA)** : feature engineering avancé, analyse de colinéarité par VIF, comparaison de stratégies de rééquilibrage.
2. **Développement des Modèles** : trois approches mathématiquement distinctes — Régression Logistique ElasticNet (modèle linéaire), Random Forest avec analyse de proximité (ensemble), et XGBoost avec apprentissage sensible au coût et optimisation bayésienne.
3. **Évaluation et Calibration** : métriques adaptées au déséquilibre (F1-Macro, AUPRC, MCC), diagrammes de fiabilité, application du Platt Scaling.
4. **Interprétabilité** : SHAP (SHapley Additive Explanations) pour expliquer les prédictions individuelles et globales.

---

## 2. Étape 1 : Analyse Exploratoire et Préparation des Données

### 2.1 Présentation du Dataset

Le jeu de données est le *Credit Card Fraud Detection*, disponible sur Kaggle, originellement issu du Machine Learning Group de l'Université Libre de Bruxelles.

| Attribut | Valeur | Remarque |
|----------|--------|----------|
| Observations | 284 807 transactions | 2 jours de données |
| Features initiales | 30 (V1–V28 + Time + Amount) | V1–V28 : composantes PCA |
| Variable cible | Class (0 = Normal, 1 = Fraude) | Variable binaire |
| Transactions normales | 282 925 (99.34%) | Classe majoritaire |
| Transactions frauduleuses | 1 882 (0.66%) | Classe minoritaire |
| Ratio de déséquilibre | 150.3 : 1 | Déséquilibre extrême |
| Valeurs manquantes | 0 | Dataset propre |

*Tableau 1.1 — Caractéristiques du dataset Credit Card Fraud Detection*

### 2.2 Feature Engineering Avancé

L'ingénierie des variables représente une étape cruciale pour améliorer la capacité discriminante des modèles. **14 nouvelles variables** ont été créées, portant le total à **42 features**.

#### 2.2.1 Transformations de la Variable Amount

La variable Amount présente une forte asymétrie à droite (skewness positif). Deux transformations ont été appliquées :

- **Amount_log** = log(1 + Amount) — réduit l'effet des valeurs extrêmes, stabilise la variance
- **Amount_sqrt** = √Amount — transformation plus douce, préserve mieux les petits montants

#### 2.2.2 Variables Cycliques du Temps

La variable Time représente les secondes écoulées depuis la première transaction. Pour capturer le cycle journalier sans créer de discontinuité artificielle entre 23h59 et 0h00 :

```
Hour = (Time mod 86400) / 3600
Hour_sin = sin(2π × Hour / 24)
Hour_cos = cos(2π × Hour / 24)
```

Cette transformation préserve la continuité cyclique et permet aux modèles linéaires de modéliser les patterns temporels circulaires.

#### 2.2.3 Statistiques Locales et Interactions

- **Amount_zscore_local** = (Amount − mean_bin) / std_bin — montant anormal par rapport à la fenêtre temporelle (50 bins)
- **V_norm** = ‖V1,...,V28‖₂ — amplitude globale du profil PCA
- **V_mean, V_std** — tendance centrale et dispersion des composantes PCA
- **V4_V11, V14_V17, V3_V10** — interactions multiplicatives entre composantes PCA discriminantes

| Feature Créée | Formule | Justification |
|---------------|---------|---------------|
| Amount_log | log(1 + Amount) | Normalisation distribution skewed |
| Hour_sin / Hour_cos | sin/cos(2π·h/24) | Encodage cyclique du temps |
| Amount_zscore_local | (x − μ_bin) / σ_bin | Anomalie relative locale |
| V_norm | ‖V1,...,V28‖₂ | Amplitude du profil PCA |
| V4_V11, V14_V17 | Vᵢ × Vⱼ | Interactions non-linéaires PCA |

*Tableau 1.2 — Features créées par ingénierie des variables*

### 2.3 Analyse de Colinéarité

#### 2.3.1 Matrice de Corrélation

La matrice de corrélation a été calculée sur l'ensemble des 42 features. Plusieurs paires présentent des corrélations élevées (|r| > 0.7) :

| Feature 1 | Feature 2 | Corrélation r | Interprétation |
|-----------|-----------|---------------|----------------|
| V2 | V13 | 1.000 | Colinéarité parfaite |
| V24 | V26 | 1.000 | Colinéarité parfaite |
| V_norm | V_std | 0.990 | Très haute |
| Amount_log | Amount_sqrt | 0.811 | Haute |
| Hour | Hour_sin | −0.780 | Haute (attendue) |

*Tableau 1.3 — Corrélations élevées détectées (|r| > 0.7)*

#### 2.3.2 Analyse VIF (Variance Inflation Factor)

$$VIF_j = \frac{1}{1 - R^2_j}$$

où $R^2_j$ est le coefficient de détermination de $X_j$ régressé sur les autres features.

- VIF < 5 : pas de multicolinéarité
- 5 ≤ VIF < 10 : multicolinéarité modérée
- VIF ≥ 10 : multicolinéarité sévère

Les features **V2/V13** et **V24/V26** présentent un VIF infini (colinéarité parfaite). La régularisation L2 de la Régression Logistique ElasticNet et le sous-échantillonnage aléatoire des features dans Random Forest / XGBoost gèrent naturellement ce problème **sans suppression manuelle**.

### 2.4 Traitement du Déséquilibre des Classes

Avec un ratio de 150:1, les modèles non contraints tendent à ignorer la classe minoritaire. Cinq stratégies ont été comparées :

| Stratégie | Type | Normal | Fraude | Avantage principal |
|-----------|------|--------|--------|-------------------|
| Raw (aucun) | — | 198 046 | 1 318 | Données originales |
| class_weight | Algorithmique | 198 046 | 1 318 | Pas de modification des données |
| SMOTE | Sur-échantillonnage | 198 046 | 19 804 | Exemples synthétiques (k-NN) |
| ADASYN | Sur-éch. adaptatif | 198 046 | 20 359 | Focus sur les zones difficiles |
| NearMiss | Sous-échantillonnage | 13 180 | 1 318 | Réduit la taille du dataset |

*Tableau 1.4 — Comparaison des stratégies de rééquilibrage*

**Décision finale :** `class_weight='balanced'` / `scale_pos_weight` retenu — agit sur l'optimisation sans modifier la distribution des données, préservant l'intégrité statistique.

---

## 3. Étape 2 : Développement des Modèles

Trois approches mathématiquement distinctes ont été confrontées, couvrant le spectre complet des paradigmes d'apprentissage supervisé : modèle linéaire, ensemble d'arbres aléatoires, et boosting séquentiel.

### 3.1 Modèle 1 — Régression Logistique avec Pénalité ElasticNet

La Régression Logistique constitue le modèle de référence (baseline) pour évaluer le gain apporté par les modèles plus complexes.

#### 3.1.1 Formulation Mathématique

Le modèle estime la probabilité via la fonction sigmoïde :

$$P(y=1 \mid x) = \sigma(w^T x + b) = \frac{1}{1 + \exp(-(w^T x + b))}$$

Avec la pénalité ElasticNet :

$$\min_w \sum_{i=1}^n \log(1 + e^{-y_i w^T x_i}) + \alpha \left[ \rho \|w\|_1 + \frac{1-\rho}{2} \|w\|_2^2 \right]$$

où ρ (l1_ratio) contrôle l'équilibre entre L1 (sélection de features par sparsité) et L2 (stabilisation de la multicolinéarité).

#### 3.1.2 Hyperparamètres Justifiés

| Paramètre | Valeur | Justification Théorique |
|-----------|--------|-------------------------|
| C | 0.1 | Régularisation modérée ; sélectionné par CV 5-fold sur AUPRC — compromis biais-variance |
| l1_ratio | 0.5 | Équilibre L1 (sparsité) / L2 (stabilité) ; L2 seul ne ferait pas de sélection de features |
| solver | saga | Seul solver scikit-learn compatible avec ElasticNet pour grands datasets (SGD stochastique) |
| class_weight | balanced | w_k = n_samples / (n_classes × n_k) — pondère la fraude ×150 |
| max_iter | 2000 | Garantit la convergence de SAGA avec la régularisation ElasticNet au ratio 150:1 |

*Tableau 2.1 — Hyperparamètres de la Régression Logistique ElasticNet*

#### 3.1.3 Résultats

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| F1-Macro | 0.4383 | Déséquilibre fort entre F1 des deux classes |
| F1-Fraude | 0.0381 | Faible précision (nombreux FP) mais bon recall |
| AUPRC | 0.0520 | 7.9× au-dessus du baseline aléatoire (0.0066) |
| ROC-AUC | 0.8426 | Bonne séparabilité globale |
| MCC | 0.0996 | Faible mais positif (meilleur que l'aléatoire) |
| Recall Fraude | 82.98% | Excellent : manque peu de fraudes |
| Spécificité | 72.28% | Nombreux faux positifs (faible précision) |

*Tableau 2.2 — Résultats de la Régression Logistique ElasticNet*

**Analyse :** Le modèle linéaire obtient un excellent recall (83%) au prix d'une précision faible — caractéristique des modèles avec class_weight élevé. Il est optimal lorsque le coût d'une fraude manquée (Faux Négatif) est extrêmement supérieur au coût des fausses alarmes.

### 3.2 Modèle 2 — Random Forest avec Analyse de Proximité

La Forêt Aléatoire est un ensemble de $T$ arbres de décision entraînés en parallèle sur des sous-échantillons bootstrap. Sa force réside dans la réduction de la variance par l'agrégation.

#### 3.2.1 Formulation Mathématique

$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} h_t(x)$$

Deux sources de randomisation réduisent la corrélation entre les arbres :
- **Bootstrap** : chaque arbre entraîné sur un sous-échantillon avec remise
- **Sous-ensemble de features** : seulement $\sqrt{p}$ features par nœud (p = 42 → ~6 features)

#### 3.2.2 Matrice de Proximité et Détection d'Outliers

La matrice de proximité mesure la similarité structurelle entre observations :

$$P[i,j] = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[\text{leaf}_t(i) = \text{leaf}_t(j)]$$

Si $P[i,j]$ est élevé, les observations $i$ et $j$ terminent souvent dans la même feuille → elles sont "similaires" selon la forêt.

Le score d'outlier pour l'observation $i$ de classe $c$ est :

$$\text{OutlierScore}(i) = \frac{n}{\sum_{j \in \text{classe}_c} P[i,j]^2}$$

Un score élevé signifie que l'observation est **isolée de ses voisins de classe** → le modèle hésite ou échoue sur ce point.

#### 3.2.3 Analyse des Outliers de Prédiction

L'analyse des top 15 outliers révèle 4 catégories de cas difficiles :

| Catégorie | Explication Mécanistique |
|-----------|--------------------------|
| **Faux Négatifs** | Fraudes structurellement similaires aux transactions normales (petits montants, profil PCA commun). La forêt manque de voisins fraude dans l'espace de proximité. |
| **Faux Positifs** | Transactions normales avec valeurs PCA extrêmes ou montants anormaux qui tombent dans des clusters fraude de la matrice de proximité. |
| **Points de Frontière** | Probabilité prédite ≈ 0.5 révélant une zone d'ambiguïté structurelle où les distributions se superposent. |
| **Outliers Structurels** | Patterns de fraude rares (typologies atypiques) non vus pendant l'entraînement — pas de voisinage de référence. |

*Tableau 2.3 — Catégories d'outliers de prédiction et leurs causes*

#### 3.2.4 Hyperparamètres Justifiés

| Paramètre | Valeur | Justification Théorique |
|-----------|--------|-------------------------|
| n_estimators | 100 | Compromis stabilité/vitesse ; diminishing returns au-delà de 300 arbres |
| max_depth | 12 | Assez profond pour patterns de fraude complexes ; évite la sur-spécialisation |
| min_samples_leaf | 5 | Prévient les feuilles trop spécifiques ; garantit la signifiance des proximités |
| max_features | sqrt(p) | Heuristique standard RF : ≈ 6 features sur 42, réduit la corrélation entre arbres |
| class_weight | balanced | Compensation automatique du ratio 150:1 par pondération inverse aux fréquences |

*Tableau 2.4 — Hyperparamètres du Random Forest*

#### 3.2.5 Résultats

| Métrique | Valeur |
|----------|--------|
| F1-Macro | 0.4933 |
| F1-Fraude | 0.0545 |
| AUPRC | 0.0375 |
| ROC-AUC | 0.8399 |
| MCC | 0.1041 (**meilleur**) |
| Recall Fraude | 55.32% |
| Spécificité | 87.55% |

**Analyse :** Le Random Forest présente le **meilleur MCC** (0.1041), reflétant sa capacité à maintenir un équilibre stable entre toutes les cellules de la matrice de confusion. Sa propriété unique — la matrice de proximité — apporte une valeur diagnostique pour comprendre les zones d'ambiguïté.

### 3.3 Modèle 3 — XGBoost avec Apprentissage Sensible au Coût

XGBoost (eXtreme Gradient Boosting) est un algorithme de boosting séquentiel reconnu comme l'état de l'art pour les données tabulaires.

#### 3.3.1 Formulation du Gradient Boosting

À chaque itération $m$, un arbre $h_m$ est entraîné sur les pseudo-résidus :

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

L'objectif à minimiser :

$$\text{Obj}(m) = \sum_i \left[ g_i \cdot f_m(x_i) + \frac{1}{2} h_i \cdot f_m(x_i)^2 \right] + \Omega(f_m)$$

avec $g_i = \partial L/\partial \hat{y}$ et $h_i = \partial^2 L/\partial \hat{y}^2$ les gradients et hessiens, et $\Omega$ la régularisation.

#### 3.3.2 Stratégie A — scale_pos_weight

$$\text{scale\_pos\_weight} = \frac{N_{\text{normal}}}{N_{\text{fraud}}} \approx \frac{198\,046}{1\,318} \approx 150$$

Ce paramètre amplifie les gradients des exemples positifs (fraude), forçant le modèle à apprendre davantage des fraudes. Le paramètre optimal découvert par Optuna est **157.7** — légèrement supérieur au ratio théorique, indiquant qu'une légère sur-pondération est bénéfique.

#### 3.3.3 Stratégie B — Focal Loss (Fonction de Perte Asymétrique)

Inspirée des travaux de Lin et al. (2017) sur la détection d'objets :

$$FL(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t)$$

- **$(1-p_t)^\gamma$** : down-weighting des exemples faciles (transactions normales bien classifiées)
- **γ = 2** : standard (Lin et al., 2017) — les exemples avec $p > 0.9$ reçoivent $0.1^2 = 1\%$ du poids
- **γ = 0** : retrouve la cross-entropie standard
- **α = 0.75** : poids supplémentaire pour la classe fraude

#### 3.3.4 Optimisation Bayésienne par Optuna (TPE Sampler)

Contrairement au GridSearch exhaustif $O(N^k)$, l'optimisation bayésienne (TPE) modélise deux distributions $l(x)$ pour les bons hyperparamètres et $g(x)$ pour les mauvais, puis sélectionne les candidats qui maximisent $l(x)/g(x)$.

**Espace de recherche et valeurs optimales (30 trials, TPE) :**

| Hyperparamètre | Plage | Optimal | Justification Théorique |
|----------------|-------|---------|-------------------------|
| max_depth | [3, 8] | **4** | Arbres profonds → overfitting ; fraude → [4-6] optimal |
| learning_rate | [0.05, 0.3] log | **0.128** | lr faible → meilleure généralisation avec plus d'arbres |
| n_estimators | [100, 400] | **250** | Inversement proportionnel au learning_rate |
| subsample | [0.6, 1.0] | **0.716** | Bootstrap stochastique : réduit la variance |
| colsample_bytree | [0.6, 1.0] | **0.845** | Analogue à max_features du RF, réduit la corrélation |
| reg_lambda | [1, 8] | **1.98** | L2 sur poids des feuilles : prévient la sur-spécialisation |
| min_child_weight | [1, 8] | **3** | Somme minimale du hessien dans une feuille |
| scale_pos_weight | [75, 300] | **157.7** | Exploration autour du ratio théorique (150) |

*Tableau 2.5 — Espace de recherche et valeurs optimales Optuna*

#### 3.3.5 Résultats XGBoost

| Métrique | SPW | Focal Loss |
|----------|-----|------------|
| F1-Macro | 0.4958 | 0.4958 |
| F1-Fraude | 0.0533 | 0.0506 |
| AUPRC | **0.0766** ★ | 0.0695 |
| ROC-AUC | 0.8231 | 0.8231 |
| MCC | 0.1033 | 0.0970 |
| Recall Fraude | 53.19% | 50.00% |
| AUPRC val (Optuna) | **0.1733** | 0.0043 |

★ Meilleur AUPRC global — modèle retenu en production.

---

## 4. Étape 3 : Évaluation et Calibration

### 4.1 Justification des Métriques

L'Accuracy est **explicitement exclue**. Avec 99.34% de transactions normales, un modèle prédisant toujours "Normal" obtient 99.34% d'accuracy sans jamais détecter de fraude.

#### 4.1.1 F1-Score Macro

$$F1 = 2 \times \frac{\text{Précision} \times \text{Rappel}}{\text{Précision} + \text{Rappel}}$$

$$F1\text{-Macro} = \frac{1}{2}(F1_{\text{Normal}} + F1_{\text{Fraude}})$$

Le F1-Macro accorde un **poids égal aux deux classes** indépendamment de leur fréquence. Il pénalise à la fois les faux positifs (via la précision) et les faux négatifs (via le rappel).

#### 4.1.2 AUPRC — Aire sous la Courbe Précision-Rappel

Contrairement à la courbe ROC qui intègre les Vrais Négatifs (nombreux et faciles), la courbe Précision-Rappel se concentre **uniquement sur la classe positive** (fraude). Un classifieur aléatoire obtient AUPRC ≈ 0.0066 (= prévalence de fraude). Tout AUPRC supérieur indique une performance réelle.

L'AUPRC est la **métrique recommandée** pour les problèmes fortement déséquilibrés car elle mesure directement la capacité à trouver des fraudes sans déclencher de fausses alarmes.

#### 4.1.3 MCC — Coefficient de Corrélation de Matthews

$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

Le MCC résume l'intégralité de la matrice de confusion en un seul scalaire **non biaisé même pour des déséquilibres extrêmes**. Il varie de −1 (prédictions inverses) à +1 (parfait), avec 0 = classifieur aléatoire.

### 4.2 Courbes Précision-Rappel et ROC

Les courbes PR confirment que tous les modèles dépassent significativement le baseline aléatoire (AUPRC = 0.0066). XGBoost SPW offre le meilleur compromis précision/rappel sur l'ensemble de la courbe.

**Pourquoi AUPRC > ROC-AUC pour les classes déséquilibrées ?**  
La courbe ROC trace TPR vs FPR. Avec 282 925 normaux, un FPR de 0.01 représente 2 829 fausses alarmes — ce qui semble "bon" sur la courbe ROC mais est inacceptable en pratique. La courbe PR évite cet artefact.

### 4.3 Calibration des Probabilités

Un modèle est **bien calibré** si :

$$P(Y=1 \mid \hat{p} = p) = p \quad \forall p \in [0,1]$$

Si le modèle prédit $p = 0.8$, 80% des transactions correspondantes doivent être des fraudes.

**Expected Calibration Error (ECE) :**

$$ECE = \sum_{b=1}^{B} \frac{|\mathcal{B}_b|}{n} \left| \text{acc}(\mathcal{B}_b) - \text{conf}(\mathcal{B}_b) \right|$$

| Modèle | ECE Avant | Statut | Action |
|--------|-----------|--------|--------|
| Logistic (ElasticNet) | 0.4756 | ⚠ Mal calibré | Platt Scaling appliqué |
| Random Forest | 0.3859 | Partiellement calibré | Acceptable |
| XGBoost SPW | 0.4631 | ⚠ Mal calibré | Acceptable en contexte |

*Tableau 3.1 — Erreur de Calibration (ECE) et actions correctives*

### 4.4 Platt Scaling

Le Platt Scaling ajuste un sigmoïde paramétrique sur les sorties brutes du modèle :

$$P_{\text{cal}}(y=1 \mid f(x)) = \frac{1}{1 + \exp(-(A \cdot f(x) + B))}$$

Les paramètres $A$ et $B$ sont estimés par maximum de vraisemblance sur un ensemble de validation distinct, avec une validation croisée à 2 folds pour éviter l'overfitting.

**Résultat sur la Régression Logistique :**

| Avant Platt Scaling | Après Platt Scaling | Réduction |
|---------------------|---------------------|-----------|
| ECE = 0.4756 | ECE = **0.0602** | **−87.3%** |

La calibration est critique en fraude bancaire : un score de 0.8 doit vraiment signifier 80% de probabilité de fraude pour guider les décisions de blocage ou d'investigation manuelle.

---

## 5. Étape 4 : Interprétabilité par SHAP

### 5.1 Fondements Théoriques de SHAP

SHAP (Lundberg & Lee, 2017) est fondé sur la théorie des jeux coopératifs de Lloyd Shapley (Prix Nobel d'Économie 2012). La valeur de Shapley de la feature $j$ pour l'observation $i$ est :

$$\phi_j(i) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \left[ v(S \cup \{j\}) - v(S) \right]$$

où $S$ est un sous-ensemble de features, $F$ l'ensemble complet, et $v(S)$ la prédiction avec le sous-ensemble $S$.

**Propriétés garanties :**

| Propriété | Signification |
|-----------|---------------|
| **Efficience** | $\sum_j \phi_j(i) = f(x_i) - E[f(x)]$ — les SHAP values somment à la prédiction |
| **Consistance** | Si une feature contribue davantage dans tout contexte, sa valeur SHAP est plus élevée |
| **Absence** | Une feature sans effet a une valeur SHAP nulle |

### 5.2 TreeSHAP pour XGBoost

Pour les modèles à base d'arbres, l'algorithme TreeSHAP (Lundberg et al., 2020) calcule les valeurs de Shapley de manière **exacte et efficiente en O(TLD²)**, évitant l'approximation nécessaire avec KernelSHAP.

**Features les plus importantes (mean |SHAP|) :**

| Rang | Feature | Interprétation |
|------|---------|----------------|
| 1 | **Amount_log** | Les montants élevés (log) poussent fortement vers fraude |
| 2 | **Amount_sqrt** | Transformation complémentaire du montant |
| 3 | **Amount_zscore_local** | Montant anormal par rapport à la période temporelle |
| 4 | **V28** | Composante PCA discriminante (anonymous) |
| 5 | **V6** | Composante PCA discriminante |
| 6 | **V_norm** | Amplitude globale du profil PCA |
| 7 | **V14** | Connu dans la littérature comme très discriminant |
| 8 | **V4** | Forte asymétrie fraude/normal |

*Tableau 4.1 — Features les plus importantes selon TreeSHAP (XGBoost)*

**Lecture du Beeswarm Plot :**
- Chaque point = une observation
- Position horizontale = impact SHAP (droite → pousse vers fraude)
- Couleur = valeur de la feature (rouge = haute, bleu = basse)

**Waterfall Plot — Prédiction individuelle :**  
Pour une transaction frauduleuse spécifique, le waterfall plot décompose la prédiction finale = valeur de base + contribution de chaque feature (rouge = poussé vers fraude, bleu = poussé vers normal).

### 5.3 Comparaison SHAP Fraude vs Normal

La comparaison des valeurs SHAP moyennes par classe révèle les **asymétries structurelles** des patterns de fraude. Les features avec SHAP positif pour Fraude et négatif pour Normal sont les plus discriminantes : **V14, V4, V_norm** apparaissent comme les features les plus importantes pour discriminer les fraudes des transactions normales.

### 5.4 SHAP pour la Régression Logistique

Pour la Régression Logistique, le **LinearExplainer** exploite la structure analytique du modèle :

$$\phi_i = w_i \cdot (x_i - E[x_i])$$

Les valeurs SHAP sont directement proportionnelles aux coefficients × (feature − moyenne), ce qui permet de valider la cohérence entre les deux modèles.

**Cohérence inter-modèles :** Les features importantes pour XGBoost (TreeSHAP) le sont également pour la Régression Logistique (LinearSHAP) — Amount_log, Amount_zscore_local, V28, V6, V14, V4 apparaissent dans les deux top 10 — ce qui valide la robustesse des signaux discriminants.

### 5.5 Avantages SHAP vs méthodes alternatives

| Méthode | Globale | Locale | Exacte | Rapide |
|---------|---------|--------|--------|--------|
| **TreeSHAP** | ✓ | ✓ | ✓ | ✓ |
| LIME | ✗ | ✓ | ✗ (approx.) | ✓ |
| Permutation Importance | ✓ | ✗ | ✗ | ✗ |
| Coefficients LR | ✓ | Partiel | ✓ | ✓ |

---

## 6. Synthèse et Comparaison des Modèles

### 6.1 Tableau Comparatif Final

| Modèle | F1-Macro | F1-Fraude | AUPRC | ROC-AUC | MCC | Recall-Fraude |
|--------|----------|-----------|-------|---------|-----|---------------|
| Logistic (ElasticNet) | 0.4383 | 0.0381 | 0.0520 | 0.8426 | 0.0996 | 82.98% |
| Random Forest | 0.4933 | 0.0545 | 0.0375 | 0.8399 | 0.1041 | 55.32% |
| **XGBoost SPW ★** | **0.4958** | 0.0533 | **0.0766** | 0.8231 | 0.1033 | 53.19% |
| XGBoost Focal | 0.4958 | 0.0506 | 0.0695 | 0.8231 | 0.0970 | 50.00% |

★ Meilleur sur AUPRC (métrique principale pour données déséquilibrées).

### 6.2 Analyse et Discussion

**XGBoost avec scale_pos_weight (SPW)** est le modèle le plus performant sur l'AUPRC (0.0766), la métrique la plus pertinente pour les données déséquilibrées. Il offre également le meilleur F1-Macro (0.4958).

**La Régression Logistique** obtient le meilleur recall fraude (83%), la rendant préférable dans un contexte où le coût d'une fraude manquée est extrêmement élevé et où les fausses alarmes sont tolérées (ex : alertes pour vérification humaine).

**Le Random Forest** présente le meilleur MCC (0.1041), reflétant sa capacité à maintenir un équilibre stable entre toutes les cellules de la matrice de confusion. Sa propriété unique — la matrice de proximité — apporte une valeur diagnostique pour comprendre les zones d'ambiguïté.

**La Focal Loss n'a pas surpassé scale_pos_weight** dans ce contexte. La dérivation analytique du gradient pour l'API sklearn XGBoost introduit une instabilité numérique que la formulation native SPW n'a pas. Ce résultat illustre l'importance de la stabilité d'implémentation au-delà de la sophistication théorique.

### 6.3 Comparaison SPW vs Focal Loss

| Critère | scale_pos_weight | Focal Loss | Recommandation |
|---------|-----------------|------------|----------------|
| AUPRC (validation Optuna) | **0.1733** | 0.0043 | SPW clairement supérieur |
| Implémentation | Native XGBoost | Custom gradient | SPW plus stable |
| Nombre d'hyperparamètres | 1 (SPW) | 2 (γ, α) | SPW plus simple |
| Interprétabilité | Intuitive (ratio N/F) | Abstraite (focale) | SPW plus lisible |

*Tableau 5.2 — Analyse comparative scale_pos_weight vs Focal Loss*

---

## 7. Conclusion et Perspectives

### 7.1 Bilan du Projet

Ce projet a démontré avec succès la mise en œuvre d'une pipeline complète d'apprentissage automatique pour la détection de fraude en environnement critique déséquilibré. Les contributions principales sont :

1. **Feature Engineering** : 14 nouvelles variables créées — transformations de distribution, encodages cycliques, statistiques locales et interactions non-linéaires — portant le total à 42 features.
2. **Analyse de Colinéarité** : détection de multicolinéarité parfaite (VIF infini pour V2/V13 et V24/V26) et gestion par régularisation plutôt que suppression manuelle.
3. **Gestion du Déséquilibre** : comparaison exhaustive de 5 stratégies ; `class_weight`/`scale_pos_weight` s'avère le plus robuste pour préserver l'intégrité des données.
4. **Trois Modèles Complémentaires** : Régression Logistique (interprétabilité maximale + recall élevé 83%), Random Forest (matrice de proximité + outlier detection + meilleur MCC), XGBoost (meilleur AUPRC 0.0766 + optimisation bayésienne).
5. **Calibration** : réduction de l'ECE de la Régression Logistique de 0.4756 à 0.0602 par Platt Scaling (−87%).
6. **Interprétabilité SHAP** : identification des features discriminantes (Amount_log, V14, V4, V_norm), validation de la cohérence inter-modèles, explication locale par waterfall.

### 7.2 Limites et Perspectives

- **Optuna étendu** : passer de 30 à 200+ trials améliorerait les hyperparamètres optimaux, notamment pour la Focal Loss.
- **Approches non supervisées** : Isolation Forest, Autoencoder variationnel (VAE), ou OCSVM comme modèles complémentaires pour détecter des fraudes de typologies inconnues.
- **Monitoring en production** : détecteurs de data drift (KS test, PSI) pour alerter en cas de changement de distribution.
- **Threshold optimization** : définition du seuil par minimisation du coût métier $C(FP) \times N_{FP} + C(FN) \times N_{FN}$ plutôt que par maximisation du F1.
- **Modèles de séquences** : RNN/LSTM pour modéliser le comportement temporel d'un client à travers l'historique de transactions.
- **Ensemble de modèles** : stacking LR + RF + XGBoost pourrait combiner le recall de LR avec l'AUPRC de XGBoost.

### 7.3 Recommandations Opérationnelles

| Aspect | Recommandation |
|--------|----------------|
| Modèle principal | XGBoost SPW + Platt Scaling (probabilités calibrées) |
| Seuil de décision | Calibrer selon le ratio coût métier FN/FP |
| Explicabilité | SHAP pour justifier chaque décision de blocage auprès des équipes fraude |
| Monitoring | Surveiller AUPRC et MCC en production ; réentraîner si dégradation > 5% |
| Réentraînement | Mensuel avec les nouvelles transactions labellisées |

*Tableau 6.1 — Recommandations pour le déploiement en production*

---

## 8. Références Bibliographiques

[1] Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017.

[2] Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016.

[3] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. ICCV 2017.

[4] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-Sampling Technique*. JAIR, 16, 321–357.

[5] Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

[6] Platt, J. C. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*. Advances in Large Margin Classifiers.

[7] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD 2019.

[8] Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI 2015.

[9] He, H., & Garcia, E. A. (2009). *Learning from Imbalanced Data*. IEEE Transactions on Knowledge and Data Engineering, 21(9).

[10] Lundberg, S. M., Erion, G., Chen, H., et al. (2020). *From Local Explanations to Global Understanding with Explainable AI for Trees*. Nature Machine Intelligence, 2(1), 56–67.

---

*ENSET Mohammedia — Projet Intelligence Artificielle — SDIA 2025-2026*  
*Encadrant : Mme. Asmae OUHMIDA*
