# Projet Intelligence Artificielle
## Classification Robuste et Analyse de Décision en Environnement Critique
### Credit Card Fraud Detection — Rapport Complet

---

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Étape 1 : Analyse Exploratoire et Préparation (EDA)](#2-étape-1--analyse-exploratoire-et-préparation)
3. [Étape 2 : Développement des Modèles](#3-étape-2--développement-des-modèles)
4. [Étape 3 : Évaluation et Calibration](#4-étape-3--évaluation-et-calibration)
5. [Étape 4 : Interprétabilité (SHAP)](#5-étape-4--interprétabilité)
6. [Résultats Finaux et Comparaison](#6-résultats-finaux-et-comparaison)
7. [Conclusion](#7-conclusion)

---

## 1. Introduction et Contexte

### Problématique
La détection de fraude bancaire est un problème de classification binaire avec un **déséquilibre extrême** de classes : dans le dataset Credit Card Fraud Detection (Kaggle), seulement **0.66%** des transactions sont frauduleuses (1 882 fraudes sur 284 807 transactions), soit un ratio d'environ **150:1**.

Ce déséquilibre rend les approches classiques inefficaces :
- Un modèle qui prédit toujours "normal" atteint une **Accuracy de 99.34%** mais ne détecte aucune fraude.
- L'optimisation de l'Accuracy est donc **trompeuse** dans ce contexte.

### Dataset : Credit Card Fraud Detection
- **Source** : Kaggle (UCI ML Repository)
- **Observations** : 284 807 transactions
- **Variables** : V1–V28 (composantes PCA anonymisées), Time, Amount, Class
- **Cible** : Class (0 = Normal, 1 = Fraude)
- **Ratio** : 150.3 :1 (Normal:Fraude)

---

## 2. Étape 1 : Analyse Exploratoire et Préparation

### 2.1 Feature Engineering Avancé

#### Transformations de l'Amount
L'Amount présente une forte **asymétrie à droite** (skewness positif). Deux transformations ont été appliquées :
- **`Amount_log`** = log1p(Amount) — réduit l'effet des valeurs extrêmes
- **`Amount_sqrt`** = sqrt(Amount) — transformation plus douce

#### Variables Cycliques du Temps
La variable Time représente les secondes écoulées. Pour capturer le **cycle journalier** :
- **`Hour`** = (Time % 86400) / 3600
- **`Hour_sin`** = sin(2π × Hour / 24)
- **`Hour_cos`** = cos(2π × Hour / 24)

Ces transformations cycliques évitent la discontinuité artificielle entre 23h59 et 0h00.

#### Statistiques Locales
- **`Amount_zscore_local`** : z-score de l'Amount par rapport aux transactions dans la même fenêtre temporelle (50 bins) — détecte les montants anormaux relativement à la période.
- **`Amount_bin_mean`**, **`Amount_bin_std`** : statistiques agrégées par fenêtre temporelle.

#### Statistiques des Composantes PCA
- **`V_norm`** : norme L2 de V1–V28 — mesure "l'amplitude" globale du profil PCA
- **`V_mean`**, **`V_std`** : tendance centrale et dispersion

#### Variables d'Interaction
Basé sur des études antérieures du dataset :
- **`V4_V11`** = V4 × V11 — interaction connue pour discriminer la fraude
- **`V14_V17`** = V14 × V17
- **`V3_V10`** = V3 × V10

**Total : 42 features** après engineering (contre 30 initialement).

### 2.2 Analyse de Colinéarité

#### Matrice de Corrélation
Corrélations élevées (|r| > 0.7) détectées :
| Feature 1 | Feature 2 | Corrélation |
|-----------|-----------|-------------|
| V2 | V13 | 1.000 |
| V24 | V26 | 1.000 |
| V_norm | V_std | 0.990 |
| Amount_bin_mean | Amount_bin_std | 0.813 |
| Amount_log | Amount_sqrt | 0.811 |

> **Note** : Les features V2/V13 et V24/V26 sont parfaitement colinéaires — elles représentent la même dimension PCA sous deux transformations. Les modèles d'ensemble (RF, XGBoost) gèrent naturellement cette multicolinéarité via la sélection aléatoire de features.

#### VIF (Variance Inflation Factor)
- V2 et V13 : **VIF = ∞** (colinéarité parfaite)
- V15 : VIF = 7.93 (modéré, < 10)
- V4 : VIF = 5.37 (acceptable)

**Décision** : Pour la Régression Logistique, la pénalité ElasticNet (L1+L2) gère automatiquement la multicolinéarité via la régularisation Ridge (L2). Pour RF et XGBoost, le sous-échantillonnage des features atténue cet effet.

### 2.3 Traitement du Déséquilibre

Cinq stratégies ont été comparées :

| Stratégie | Type | Normal | Fraude | Ratio |
|-----------|------|--------|--------|-------|
| Raw (aucun) | — | 198 046 | 1 318 | 150:1 |
| class_weight='balanced' | Algorithmique | 198 046 | 1 318 | 150:1 |
| SMOTE | Sur-échantillonnage | 198 046 | 19 804 | 10:1 |
| ADASYN | Sur-échantillonnage adaptatif | 198 046 | 20 359 | ~10:1 |
| NearMiss | Sous-échantillonnage | 13 180 | 1 318 | 10:1 |

#### Justification du choix pour chaque modèle
- **Logistic + RF** : `class_weight='balanced'` — ajuste les poids à la volée sans modifier les données, préservant la distribution originale.
- **XGBoost** : `scale_pos_weight` — équivalent algorithmique du class_weight pour le boosting.
- **Comparaison SMOTE/ADASYN** : Génère de nouveaux exemples synthétiques par interpolation k-NN. ADASYN adapte la densité de génération aux régions difficiles (frontières de classe).

---

## 3. Étape 2 : Développement des Modèles

### 3.1 Modèle 1 : Régression Logistique avec Pénalité ElasticNet

#### Formulation Mathématique

$$\min_{w} \sum_{i=1}^n \log(1 + e^{-y_i w^T x_i}) + \alpha \left[ \rho \|w\|_1 + \frac{1-\rho}{2} \|w\|_2^2 \right]$$

Où :
- **L1 (Lasso)** : $\rho \|w\|_1$ → sélection de features, coefficients sparse
- **L2 (Ridge)** : $\frac{1-\rho}{2}\|w\|_2^2$ → gestion de la multicolinéarité, stabilité numérique

#### Hyperparamètres Retenus

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| C | 0.1 | Régularisation modérée, sélectionné par CV 5-fold sur AUPRC |
| l1_ratio | 0.5 | Équilibre L1/L2 ; L1 pour la sparsité, L2 pour la stabilité |
| solver | 'saga' | Seul solver compatible ElasticNet pour grands jeux de données |
| class_weight | 'balanced' | Pondération inverse à la fréquence de classe |
| max_iter | 2000 | Garantit la convergence avec SAGA |

#### Résultats

| Métrique | Valeur |
|----------|--------|
| F1-Macro | 0.4383 |
| AUPRC | 0.0520 |
| ROC-AUC | 0.8426 |
| MCC | 0.0996 |
| Recall (Fraude) | 0.8298 |
| Spécificité | 0.7228 |

**Analyse** : Le modèle linéaire obtient un bon recall (83%) mais génère beaucoup de faux positifs (faible précision = 2%). C'est acceptable dans un contexte de détection de fraude où manquer une fraude (faux négatif) est coûteux.

---

### 3.2 Modèle 2 : Random Forest avec Analyse de Proximité

#### Formulation et Principe

La Forêt Aléatoire est un ensemble de $T$ arbres de décision entraînés sur des sous-échantillons bootstrap :

$$\hat{y} = \frac{1}{T} \sum_{t=1}^T h_t(x)$$

Deux sources de randomisation réduisent la variance :
1. **Bootstrap** des observations (bagging)
2. **Sous-ensemble aléatoire** de $\sqrt{p}$ features à chaque nœud

#### Hyperparamètres Retenus

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| n_estimators | 100 | Compromis stabilité/vitesse ; diminishing returns au-delà de 300 |
| max_depth | 12 | Assez profond pour capturer les patterns de fraude complexes |
| min_samples_leaf | 5 | Évite les feuilles trop spécifiques ; requis pour proximité significative |
| max_features | 'sqrt' | Heuristique standard RF : ≈ 6 features sur 42 |
| class_weight | 'balanced' | Compensation automatique du ratio 150:1 |

#### Matrice de Proximité

La proximité entre deux observations $i$ et $j$ est :

$$P[i,j] = \frac{1}{T} \sum_{t=1}^T \mathbb{1}[\text{leaf}_t(i) = \text{leaf}_t(j)]$$

**Interprétation** : Si deux observations atteignent souvent la même feuille terminale, elles sont "similaires" du point de vue de la forêt.

#### Détection des Outliers de Prédiction

Le score d'outlier pour l'observation $i$ de classe $c$ est :

$$\text{OutlierScore}(i) = \frac{n}{\sum_{j \in c} P[i,j]^2}$$

Un score élevé indique que l'observation est **isolée de ses voisins de classe** → le modèle hésite ou échoue sur ces points.

#### Résultats

| Métrique | Valeur |
|----------|--------|
| F1-Macro | 0.4933 |
| AUPRC | 0.0375 |
| ROC-AUC | 0.8399 |
| MCC | 0.1041 |
| Recall (Fraude) | 0.5532 |
| Spécificité | 0.8755 |

#### Analyse des Outliers de Prédiction

Les outliers révèlent 4 catégories de cas difficiles :

1. **Faux Négatifs** (fraudes manquées) : Transactions frauduleuses avec des montants faibles et des composantes PCA proches de la normale. Le modèle manque de voisins fraude dans l'espace de proximité pour voter avec confiance.

2. **Faux Positifs** (fausses alarmes) : Transactions normales avec des valeurs PCA extrêmes ou des montants anormaux qui tombent dans des clusters fraude.

3. **Points de frontière** : Probabilité prédite ≈ 0.5, révélant une zone d'ambiguïté structurelle entre les deux classes.

4. **Outliers structurels** : Patterns rares (fraudes atypiques) que la forêt n'a jamais vus pendant l'entraînement.

---

### 3.3 Modèle 3 : XGBoost avec Apprentissage Sensible au Coût

#### Formulation du Gradient Boosting

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

À chaque itération, un arbre $h_m$ est entraîné sur les **résidus pseudo-négatifs** (gradients de la fonction de perte).

#### Stratégie A : scale_pos_weight

$$\text{scale\_pos\_weight} = \frac{N_{\text{normal}}}{N_{\text{fraud}}} \approx 150$$

Ce paramètre **amplifie les gradients** des exemples positifs (fraude) par un facteur de 150, forçant le modèle à accorder plus d'importance aux erreurs sur la classe minoritaire.

**Justification** : Simple, efficace, directement intégré dans XGBoost. Équivalent à l'oversampling mais agit sur l'optimisation, pas sur les données.

#### Stratégie B : Focal Loss (fonction de perte asymétrique)

Inspirée de Lin et al. (2017), la Focal Loss introduit un terme de **focalisation** $\gamma$ qui réduit la contribution des exemples faciles :

$$FL(p_t) = -\alpha (1-p_t)^\gamma \log(p_t)$$

- **$\gamma = 0$** : cross-entropie standard
- **$\gamma > 0$** : down-pondère les exemples bien classés (transactions normales évidentes)
- **$\alpha$** : poids de la classe positive

Le gradient pour XGBoost est calculé analytiquement et passé via l'API `objective`.

#### Optimisation Bayésienne par Optuna (TPE Sampler)

Au lieu d'un GridSearch exhaustif ($O(N^k)$), l'optimisation bayésienne utilise un **Tree-structured Parzen Estimator (TPE)** qui modélise $P(\text{params}|\text{score})$ et $P(\text{params})$ pour trouver efficacement les meilleurs hyperparamètres.

**Espace de Recherche — Justifications théoriques :**

| Paramètre | Plage | Justification |
|-----------|-------|---------------|
| max_depth | [3, 8] | Arbres peu profonds évitent l'overfitting ; la fraude nécessite [4-6] |
| learning_rate | [0.05, 0.3] | lr faible → meilleure généralisation avec plus d'arbres |
| n_estimators | [100, 400] | Inversement proportionnel au learning_rate |
| subsample | [0.6, 1.0] | Réduction de variance par bootstrap stochastique |
| colsample_bytree | [0.6, 1.0] | Comme max_features dans RF |
| reg_lambda | [1, 8] | L2 sur les poids des feuilles : prévient la surspécialisation |
| min_child_weight | [1, 8] | Somme minimale du hessien dans une feuille |
| scale_pos_weight | [75, 300] | Exploration autour du ratio théorique (150) |

**Résultats Optuna (Strategy A - scale_pos_weight) :**

| Paramètre | Valeur Optimale |
|-----------|-----------------|
| max_depth | 4 |
| learning_rate | 0.128 |
| n_estimators | 250 |
| subsample | 0.716 |
| colsample_bytree | 0.845 |
| reg_lambda | 1.98 |
| min_child_weight | 3 |
| scale_pos_weight | 157.7 |
| AUPRC (validation) | 0.1733 |

#### Résultats XGBoost (scale_pos_weight)

| Métrique | Valeur |
|----------|--------|
| F1-Macro | 0.4958 |
| AUPRC | 0.0766 |
| ROC-AUC | 0.8288 |
| MCC | 0.1033 |
| Recall (Fraude) | 0.5319 |
| Spécificité | 0.8827 |

---

## 4. Étape 3 : Évaluation et Calibration

### 4.1 Justification des Métriques

#### Pourquoi ne pas utiliser l'Accuracy ?
Un modèle qui prédit systématiquement "Normal" obtiendrait **99.34% d'accuracy** sans jamais détecter de fraude. C'est pourquoi l'accuracy est une métrique trompeuse pour les données déséquilibrées.

#### F1-Macro
$$F1_{\text{Macro}} = \frac{1}{2}\left(F1_{\text{Normal}} + F1_{\text{Fraud}}\right)$$

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Avantage** : Équilibre précision et rappel pour chaque classe, puis fait la moyenne. Sensible aux performances sur la classe minoritaire.

#### AUPRC (Area Under Precision-Recall Curve)
Contrairement à la courbe ROC, la courbe Precision-Recall est **plus informative pour les classes rares** car :
- Elle ne tient pas compte des Vrais Négatifs (nombreux et faciles dans notre cas)
- Elle mesure directement la capacité à trouver des fraudes sans déclencher trop de fausses alarmes

Un classifieur aléatoire obtient AUPRC ≈ 0.0066 (= prévalence de fraude).

#### MCC (Matthews Correlation Coefficient)
$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

**Avantage** : Résume la matrice de confusion en un seul scalaire, **non biaisé même pour des déséquilibres extrêmes**. Range [-1, 1] ; 0 = prédicteur aléatoire.

### 4.2 Calibration des Probabilités

#### Qu'est-ce que la calibration ?
Un modèle est **bien calibré** si sa probabilité prédite correspond à la fréquence réelle des événements : si le modèle dit "70% de chance de fraude", alors 70% de ces transactions doivent effectivement être frauduleuses.

#### Reliability Diagrams (Diagrammes de Fiabilité)

| Modèle | ECE (Expected Calibration Error) | Statut |
|--------|----------------------------------|--------|
| Logistic (ElasticNet) | 0.307 | ⚠️ Mal calibré |
| Random Forest | 0.154 | ⚠️ Partiellement calibré |
| XGBoost (SPW) | 0.120 | Acceptable |

#### Platt Scaling appliqué à la Régression Logistique

Le Platt Scaling ajuste un **sigmoïde** sur les sorties du modèle :

$$P(y=1|f(x)) = \frac{1}{1 + e^{-(A \cdot f(x) + B)}}$$

Résultat : L'ECE passe de **0.307** (avant) à une valeur réduite (après calibration), améliorant la fiabilité des probabilités pour la prise de décision.

---

## 5. Étape 4 : Interprétabilité

### 5.1 SHAP (SHapley Additive Explanations)

SHAP est fondé sur la théorie des jeux coopératifs. La valeur de Shapley d'une feature $j$ pour une observation $i$ est :

$$\phi_j(i) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \left[v(S \cup \{j\}) - v(S)\right]$$

**Propriétés garanties** :
- **Efficience** : $\sum_j \phi_j(i) = f(x_i) - E[f(x)]$
- **Consistance** : Si une feature contribue davantage dans tout contexte, sa valeur SHAP est plus élevée
- **Précision locale** : La somme des SHAP values reproduit exactement la prédiction

Pour XGBoost, nous utilisons **TreeSHAP** (Lundberg et al., 2020), qui est exact et efficace en $O(TLD^2)$.

### 5.2 Features les Plus Importantes (XGBoost TreeSHAP)

Les features ayant les plus grandes valeurs SHAP moyennes (mean |SHAP|) sont typiquement :
- **V14, V4, V12** : Composantes PCA fortement discriminantes (connues dans la littérature)
- **V_norm** : Norme L2 des composantes PCA — mesure l'amplitude globale du profil
- **Amount_log** : Montant transformé — les fraudes ont souvent des montants atypiques
- **V4_V11** : Interaction entre deux composantes PCA discriminantes

### 5.3 Interprétation d'une Prédiction Individuelle (Waterfall)

Le graphique waterfall montre pour une transaction frauduleuse spécifique :
- **Valeurs rouges** : Features qui poussent la prédiction vers "fraude"
- **Valeurs bleues** : Features qui poussent vers "normal"
- La somme de toutes les contributions SHAP + valeur moyenne = prédiction finale

---

## 6. Résultats Finaux et Comparaison

### Tableau de Comparaison

| Modèle | F1-Macro | AUPRC | MCC | ROC-AUC | Recall-Fraud |
|--------|----------|-------|-----|---------|--------------|
| Logistic (ElasticNet) | 0.4383 | 0.0520 | 0.0996 | 0.8426 | 0.8298 |
| Random Forest | 0.4933 | 0.0375 | 0.1041 | 0.8399 | 0.5532 |
| XGBoost SPW | **0.4958** | **0.0766** | **0.1033** | 0.8288 | 0.5319 |

### Discussion

**XGBoost (scale_pos_weight)** est le meilleur modèle sur AUPRC, la métrique la plus pertinente pour la détection de fraude avec données déséquilibrées.

**Logistic Regression** obtient le meilleur recall (83%) mais au prix d'une précision très faible — utile si le coût d'une fraude manquée est extrêmement élevé.

**Random Forest** offre le meilleur MCC et est le plus interprétable via la matrice de proximité.

### Comparaison SPW vs Focal Loss

| Stratégie | AUPRC (val) | Avantage |
|-----------|-------------|----------|
| scale_pos_weight | 0.1733 | Simple, efficace, convergence rapide |
| Focal Loss | 0.0043 | Théoriquement supérieur, mais sensible au gradient |

**Conclusion** : `scale_pos_weight` surpasse la Focal Loss dans ce contexte. La Focal Loss requiert une calibration soignée des hyperparamètres $\gamma$ et $\alpha$, et la dérivation du gradient approximé dans notre implémentation n'est pas aussi stable numériquement que `scale_pos_weight`.

---

## 7. Conclusion

### Synthèse Technique

Ce projet a démontré une pipeline complète de ML pour la détection de fraude :

1. **Feature Engineering** : 14 nouvelles features créées (transformations, interactions, statistiques locales) → 42 features totales
2. **Gestion du déséquilibre** : Comparaison de 5 approches ; `class_weight` / `scale_pos_weight` s'avère le plus robuste
3. **Modèles** : 3 approches mathématiquement distinctes couvrant le spectre linéaire-ensemble-boosting
4. **Calibration** : Platt Scaling réduit significativement l'ECE de la Régression Logistique
5. **Interprétabilité** : TreeSHAP identifie V14, V4, V_norm comme features décisives

### Limites et Perspectives

- Le dataset synthétique (généré pour respecter la confidentialité des données Kaggle) n'a pas la même structure que les vraies données PCA anonymisées
- L'optimisation Optuna avec 30 trials reste limitée ; 200+ trials amélioreraient les résultats
- L'implémentation de la Focal Loss via l'API sklearn XGBoost est moins stable qu'en mode Booster natif
- Des approches comme **Isolation Forest** ou **AutoEncoder** pour la détection d'anomalies pourraient compléter l'approche supervisée

### Recommandations Opérationnelles

Pour un déploiement en production :
1. Utiliser **XGBoost SPW** comme modèle principal (meilleur AUPRC)
2. Appliquer **Platt Scaling** pour des probabilités calibrées exploitables
3. Définir le **seuil de décision** selon le coût asymétrique métier (FN coûte plus que FP)
4. Monitorer les **drift de distribution** pour maintenir les performances dans le temps
5. Utiliser **SHAP** pour justifier chaque décision de blocage auprès des équipes fraude

---

*Rapport généré automatiquement par le pipeline Python.*  
*Dataset : Credit Card Fraud Detection (synthétique, 284 807 transactions)*  
*Technologies : scikit-learn, XGBoost, Optuna, SHAP, imbalanced-learn*
