# Sales Forecasting Retail

Pipeline de prévision des ventes par produit (SKU) et agence sur un horizon de **4 mois**.

## Table des matières

- [Contexte](#contexte)
- [Approche Méthodologique](#approche-méthodologique)
- [Architecture](#architecture)
- [Installation](#installation)
- [Données](#données)
- [Features Incluses](#features-incluses)
- [Usage](#usage)
- [Modèles](#modèles)
- [Métriques](#métriques)
- [Vision Production (GCP)](#vision-production-gcp)
- [Limitations et Améliorations](#limitations-et-améliorations)

## Contexte

Ce projet répond à un besoin de prévision des volumes de ventes mensuels pour optimiser :

- La gestion des stocks
- La planification logistique
- Les décisions commerciales

## Approche Méthodologique

Ce projet représente la **vision production** d'une solution de prévision. En amont, le workflow data science classique a été suivi :

1. **Business Understanding** : Comprendre le besoin métier (prévision 4 mois, granularité SKU/agence)
2. **EDA** : Analyse exploratoire pour comprendre les données, saisonnalités, tendances
3. **Feature Engineering** : Identification des features pertinentes (lags, rolling means)
4. **Modelling** : Comparaison de plusieurs algorithmes (RF, XGBoost, LightGBM)
5. **Evaluation** : Sélection du meilleur modèle sur métriques métier

Le code présent se concentre sur **l'industrialisation** : code modulaire, reproductible et déployable.

## Architecture

```text
sales-forecasting-retail/
├── src/
│   ├── configs/
│   │   └── config.py          # Configuration centralisée
│   ├── data/
│   │   ├── loader.py          # Chargement des données
│   │   └── preprocessing.py   # Préparation et encodage
│   ├── features/
│   │   └── engineering.py     # Feature engineering
│   ├── model/
│   │   ├── training.py        # Entraînement des modèles
│   │   └── evaluation.py      # Métriques d'évaluation
│   └── inference/
│       └── predictor.py       # Prédiction
├── scripts/
│   ├── train.py               # Script d'entraînement
│   └── predict.py             # Script de prédiction
├── data/
│   ├── raw/                   # Données brutes
│   └── processed/             # Données transformées
├── models/                    # Modèles sauvegardés
└── outputs/                   # Prédictions générées
```

## Installation

```bash
# Cloner le repo
git clone https://github.com/jass228/sales-forecasting-retail.git
cd sales-forecasting-retail

# Option 1 : avec uv (recommandé)
uv sync

# Option 2 : avec pip
pip install -e .
```

## Données

Le dataset contient les colonnes suivantes :

| Colonne                            | Description                     | Utilisée                |
| ---------------------------------- | ------------------------------- | ----------------------- |
| `agency`                           | Identifiant de l'agence/magasin | ✅ Encodée              |
| `sku`                              | Identifiant du produit          | ✅ Encodée              |
| `date`                             | Date (granularité mensuelle)    | ✅ Features temporelles |
| `volume`                           | Volume des ventes (cible)       | ✅ Target + lags        |
| `avg_max_temp`                     | Température moyenne maximale    | ✅ Exogène              |
| `price_actual`                     | Prix actuel                     | ✅ Exogène              |
| `price_regular`                    | Prix régulier                   | ✅ Exogène              |
| `discount`                         | Montant de la remise            | ✅ Exogène              |
| `discount_in_percent`              | Pourcentage de remise           | ✅ Exogène              |
| `industry_volume`                  | Volume de l'industrie           | ✅ Exogène              |
| `soda_volume`                      | Volume de sodas                 | ✅ Exogène              |
| `avg_population_2017`              | Population moyenne              | ✅ Exogène              |
| `avg_yearly_household_income_2017` | Revenu moyen des ménages        | ✅ Exogène              |
| Colonnes événements                | Jours fériés et événements      | ✅ Exogène              |

**Note sur les features exogènes à l'inférence** : Si ces colonnes ne sont pas fournies dans `new_data.csv`, elles sont automatiquement remplies avec les **dernières valeurs connues** de l'historique pour chaque couple agency/sku.

## Features Incluses

Le modèle utilise **32 features** au total.

**Logique de sélection des features** :

- **Lags et rolling means** : Capturent l'autocorrélation et les tendances des ventes passées (features les plus prédictives pour les séries temporelles)
- **Features temporelles** : Capturent la saisonnalité (mois, trimestre)
- **Features exogènes** : Toutes les variables du dataset sont incluses car elles peuvent influencer les ventes (prix, météo, événements)

En production, une analyse d'importance des features permettrait d'identifier et supprimer les features peu contributives.

### Features catégorielles

- `agency`, `sku` : Encodées avec OrdinalEncoder

### Features temporelles (extraites de `date`)

- `year`, `month`, `quarter`

### Features de lag (historique des ventes)

- `volume_lag_1`, `volume_lag_2`, `volume_lag_3`, `volume_lag_6`, `volume_lag_12`

### Features de rolling mean (moyennes mobiles)

- `volume_rolling_mean_3`, `volume_rolling_mean_6`, `volume_rolling_mean_12`

### Features exogènes

- **Prix** : `price_actual`, `price_regular`, `discount`, `discount_in_percent`
- **Marché** : `industry_volume`, `soda_volume`
- **Météo** : `avg_max_temp`
- **Démographie** : `avg_population_2017`, `avg_yearly_household_income_2017`

### Features événements

- `easter_day`, `good_friday`, `new_year`, `christmas`, `labor_day`, `independence_day`, `revolution_day_memorial`, `regional_games`, `beer_capital`, `music_fest`

## Usage

### Entraînement

```bash
# Avec uv
uv run python scripts/train.py

# Avec pip
python scripts/train.py
```

**Artefacts générés** :

| Fichier                       | Description                                             |
| ----------------------------- | ------------------------------------------------------- |
| `models/best_model.pkl`       | Meilleur modèle entraîné                                |
| `models/encoder.pkl`          | Encoder pour agency/sku                                 |
| `models/history.csv`          | 12 derniers mois (pour calculer les lags à l'inférence) |
| `models/model_comparison.csv` | Comparaison des performances des 3 modèles              |
| `data/processed/`             | Données préparées (train, test)                         |

### Prédiction

```bash
# Placer les nouvelles données dans data/raw/new_data.csv

# Avec uv
uv run python scripts/predict.py

# Avec pip
python scripts/predict.py
```

**Format minimal de `new_data.csv`** :

```csv
date,agency,sku
2018-01-01,Agency_01,SKU_01
2018-02-01,Agency_01,SKU_01
...
```

Les colonnes exogènes manquantes sont automatiquement remplies avec les dernières valeurs connues.

**Sortie** : `outputs/predictions.csv`

## Modèles

### Choix des algorithmes

Trois modèles tree-based sont comparés :

| Modèle           | Configuration            |
| ---------------- | ------------------------ |
| **RandomForest** | 300 arbres, max_depth=12 |
| **XGBoost**      | 500 estimateurs, lr=0.05 |
| **LightGBM**     | 500 estimateurs, lr=0.05 |

**Pourquoi des modèles tree-based ?**

- **Performants sur données tabulaires** : Les arbres de décision excellent sur ce type de données (features mixtes numériques/catégorielles)
- **Pas de normalisation requise** : Contrairement aux réseaux de neurones ou à la régression linéaire
- **Gestion native des non-linéarités** : Capturent les interactions complexes entre features
- **Interprétables** : Feature importance disponible pour expliquer les prédictions

Le meilleur modèle est sélectionné automatiquement sur le MAE.

> **Note** : Les hyperparamètres utilisés sont des valeurs par défaut raisonnables. L'objectif de ce POC est de démontrer une architecture industrialisable, pas d'optimiser les performances du modèle. En production, une étape d'optimisation serait ajoutée.

## Métriques

| Métrique | Description                                  |
| -------- | -------------------------------------------- |
| **MAE**  | Erreur moyenne absolue (métrique principale) |
| **RMSE** | Pénalise les grosses erreurs                 |
| **MAPE** | Erreur en pourcentage                        |

**Pourquoi MAE comme métrique principale ?**

- **Interprétable** : Une MAE de 250 signifie qu'on se trompe en moyenne de 250 unités de volume
- **Robuste aux outliers** : Contrairement au RMSE, le MAE ne sur-pénalise pas les erreurs extrêmes
- **Adapté au métier** : Pour la gestion des stocks, une erreur moyenne constante est plus facile à anticiper qu'une erreur variable

---

## Vision Production (GCP)

Cette section décrit comment cette solution serait déployée et opérée sur Google Cloud Platform.

### Architecture Vertex AI

```text
┌────────────────────────────────────────────────────────────────────────────────┐
│                              VERTEX AI PIPELINES                               │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  PIPELINE TRAINING                                                             │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐             │
│  │  BigQuery  │──►│  Feature   │──►│  Training  │──►│   Model    │             │
│  │  (source)  │   │ Engineering│   │    Job     │   │  Registry  │             │
│  └────────────┘   └────────────┘   └────────────┘   └─────┬──────┘             │
│                                                           │                    │
│  PIPELINE INFERENCE                                       │                    │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐         │                    │
│  │  BigQuery  │──►│  Feature   │──►│   Batch    │◄────────┘                    │
│  │ (new data) │   │ Engineering│   │ Prediction │                              │
│  └────────────┘   └────────────┘   └─────┬──────┘                              │
│                                          │                                     │
│                                          ▼                                     │
│                                   ┌────────────┐                               │
│                                   │  BigQuery  │                               │
│                                   │(résultats) │                               │
│                                   └────────────┘                               │
└────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                                   MONITORING                                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐               │
│  │   Vertex AI     │──►│ Cloud Monitoring│──►│  Cloud Alerting │──► Slack      │
│  │ Model Monitoring│   │   (dashboards)  │   │                 │               │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘               │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Composants

| Composant                      | Rôle                                                            |
| ------------------------------ | --------------------------------------------------------------- |
| **BigQuery**                   | Stockage des données source (si déjà présent dans l'écosystème) |
| **Vertex AI Training**         | Custom job pour l'entraînement du modèle                        |
| **Vertex AI Model Registry**   | Versioning des modèles                                          |
| **Vertex AI Batch Prediction** | Inférence en batch (voir stratégie ci-dessous)                  |
| **Vertex AI Pipelines**        | Orchestration avec Kubeflow                                     |
| **Vertex AI Feature Store**    | Versioning des features (optionnel)                             |
| **Cloud Monitoring**           | Alerting et dashboards                                          |

### Stratégie d'Inférence

Pour ce cas d'usage (prévision mensuelle de ventes), une **inférence batch** est recommandée plutôt qu'un endpoint temps réel :

| Critère            | Batch                | Temps réel                 |
| ------------------ | -------------------- | -------------------------- |
| Fréquence besoin   | Hebdomadaire/mensuel | À la demande               |
| Latence acceptable | Minutes/heures       | Millisecondes              |
| Coût               | Faible (pay-per-use) | Élevé (endpoint 24/7)      |
| Complexité         | Simple               | Infrastructure à maintenir |

**Recommandation** : Vertex AI Batch Prediction avec scheduling hebdomadaire via Vertex AI Pipelines.

### Stratégie de Monitoring

#### 1. Monitoring des performances

| Métrique      | Seuil d'alerte      | Action         |
| ------------- | ------------------- | -------------- |
| MAE en prod   | > 1.5x MAE baseline | Investigation  |
| MAPE          | > 20%               | Réentraînement |
| Latency batch | > 2h                | Scaling        |

> **Note** : Ces seuils sont des exemples. En production, ils seraient définis avec les équipes métier en fonction de l'impact business.

#### 2. Détection du drift

| Type de drift        | Méthode                                            | Fréquence      |
| -------------------- | -------------------------------------------------- | -------------- |
| **Data drift**       | KS-test sur distributions des features             | Hebdomadaire   |
| **Concept drift**    | Monitoring MAE sur données récentes vs historiques | Hebdomadaire   |
| **Prediction drift** | Distribution des prédictions vs baseline           | À chaque batch |

**Implémentation** :

- Vertex AI Model Monitoring pour le data drift automatique
- Custom metrics dans Cloud Monitoring pour le concept drift
- Alerting via Cloud Alerting + notification Slack/email

### Stratégie de Réentraînement

| Trigger               | Condition                      | Action                         |
| --------------------- | ------------------------------ | ------------------------------ |
| **Scheduled**         | Mensuel                        | Réentraînement automatique     |
| **Performance-based** | MAE > seuil pendant 2 semaines | Réentraînement déclenché       |
| **Data-based**        | Drift détecté (p-value < 0.05) | Réentraînement + investigation |
| **Manuel**            | Changement business majeur     | Réentraînement + revalidation  |

> **Note** : Les fréquences et conditions sont des exemples. En production, elles seraient définies selon le contexte métier et la vitesse d'évolution des données.

**Pipeline de réentraînement** :

1. Récupération des nouvelles données (BigQuery)
2. Feature engineering
3. Entraînement sur fenêtre glissante (ex: 24 derniers mois)
4. Évaluation sur données récentes non vues par le modèle
5. Comparaison avec modèle en prod
6. Déploiement si amélioration > seuil

---

## Limitations et Améliorations

### Limitations actuelles

| Limitation                          | Impact                         | Compromis réalisé                                                            |
| ----------------------------------- | ------------------------------ | ---------------------------------------------------------------------------- |
| **Cold start**                      | Nouveaux SKU/agences non gérés | Nécessite historique minimum de 12 mois                                      |
| **Hyperparamètres fixes**           | Performance non optimale       | Priorité donnée à l'architecture vs tuning                                   |
| **Pas de tests unitaires**          | Maintenabilité réduite         | Contrainte de temps                                                          |
| **Horizon unique**                  | Même modèle pour H+1 à H+4     | Simplification acceptable pour POC                                           |
| **Features exogènes à l'inférence** | Approximation                  | Remplies avec dernières valeurs connues (hypothèse de stabilité court terme) |

### Améliorations possibles

- **Tests unitaires et d'intégration** : pytest sur les modules critiques
- **Logging structuré** : Traçabilité des runs (logging Python ou MLflow)
- **Hyperparameter tuning** : Optimisation des hyperparamètres avec cross-validation temporelle
- **Cross-validation temporelle** : TimeSeriesSplit pour une évaluation plus robuste
- **Feature selection** : Analyse de corrélation, importance des features, suppression des features redondantes
- **Multi-horizon** : Un modèle spécialisé par horizon (H+1, H+2, H+3, H+4)
- **Interprétabilité** : SHAP values pour expliquer les prédictions au métier
- **Cold start** : Modèle hiérarchique pour les nouveaux produits/agences
- **Containerisation** : Docker + CI/CD

## Auteur

Joseph A.

## License

Ce projet est sous licence [MIT](LICENSE).
