# Sales Forecasting Retail

Pipeline de prévision des ventes par produit (SKU) et agence sur un horizon de 15 jours.

## 📋 Contexte

Ce projet répond à un besoin de prévision des volumes de ventes mensuels pour optimiser :

- La gestion des stocks
- La planification logistique
- Les décisions commerciales

## 🏗️ Architecture

```
sales-forecasting-retail/
├── src/
│   ├── data/
│   │   └── loader.py           # Chargement et split des données
│   ├── features/
│   │   └── engineering.py      # Feature engineering
│   ├── models/
│   │   └── trainer.py          # Entraînement LightGBM
│   ├── evaluation/
│   │   └── metrics.py          # Métriques (MAE, RMSE, MAPE)
│   └── inference/
│       └── predictor.py        # Prédiction et sauvegarde
├── scripts/
│   ├── train.py                # Script d'entraînement
│   └── predict.py              # Script de prédiction
├── data/
│   └── raw/                    # Données brutes
├── models/                     # Modèles sauvegardés
└── outputs/                    # Prédictions générées
```

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/jass228/sales-forecasting-retail.git
cd sales-forecasting-retail

# Installer les dépendances avec uv
uv sync
```

## 📊 Données

Le dataset contient les colonnes suivantes :

- `agency` : Identifiant de l'agence
- `sku` : Identifiant du produit
- `date` : Date (granularité mensuelle)
- `volume` : Volume des ventes (cible)
- `avg_max_temp` : Température moyenne maximale
- `price_actual` : Prix actuel
- `discount_in_percent` : Pourcentage de remise
- Features calendaires (jours fériés, événements)

## 🎯 Usage

### Entraînement

```bash
uv run python scripts/train.py --data data/raw/ds_assortiment_dataset.csv
```

Options :

- `--test-date` : Date de début du test set (défaut: 2017-07-01)
- `--model-output` : Chemin du modèle (défaut: models/model.pkl)
- `--artifacts-output` : Chemin des artifacts (défaut: models/artifacts.pkl)

### Prédiction

```bash
# Sur des données existantes
uv run python scripts/predict.py --data data/raw/new_data.csv

# Prévisions sur dates futures
uv run python scripts/predict.py --forecast --start-date 2018-01-01 --end-date 2018-03-01
```

Options :

- `--model` : Chemin du modèle (défaut: models/model.pkl)
- `--artifacts` : Chemin des artifacts (défaut: models/artifacts.pkl)
- `--output` : Chemin de sortie (défaut: outputs/predictions.csv)

## 🔧 Features Engineering

### Features temporelles

- `year`, `month`, `quarter`
- `day_of_month`, `day_of_week`, `week_of_year`

### Features historiques

- `mean_volume_agency_sku_month` : Moyenne par agence/SKU/mois
- `mean_volume_agency_sku` : Moyenne par agence/SKU
- `mean_volume_sku_month` : Moyenne par SKU/mois (saisonnalité produit)

### Features du dataset

- `avg_max_temp` : Impact météo
- `price_actual` : Impact prix
- `discount_in_percent` : Impact promotions

## 📈 Modèle

**Algorithme** : LightGBM (Gradient Boosting)

**Choix justifié** :

- Performant sur données tabulaires
- Gère nativement les variables catégorielles
- Rapide à entraîner
- Interprétable (feature importance)

**Alternatives considérées** :

- XGBoost : Performances similaires, temps d'entraînement plus long
- Prophet : Adapté aux séries temporelles univariées, moins flexible
- LSTM : Overkill pour des données mensuelles (~60 points par série)

## 📉 Métriques

| Métrique | Description                                            |
| -------- | ------------------------------------------------------ |
| MAE      | Mean Absolute Error - Erreur moyenne en valeur absolue |
| RMSE     | Root Mean Squared Error - Pénalise les grosses erreurs |
| MAPE     | Mean Absolute Percentage Error - Erreur en pourcentage |

### Baseline

Moyenne historique par agence/SKU/mois. Le modèle doit battre cette baseline pour être considéré utile.

## 🏭 Vision Production (GCP)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cloud Composer                           │
│                    (Orchestration Airflow)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │  BigQuery   │     │  Vertex AI  │     │ Cloud Run   │
   │  (Data)     │     │ (Training)  │     │ (API)       │
   └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                   │
          └───────────────────┴───────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Cloud Storage     │
                    │ (Models/Artifacts)│
                    └───────────────────┘
```

### Composants

- **Cloud Composer** : Orchestration des pipelines (DAGs Airflow)
- **BigQuery** : Stockage et requêtage des données
- **Vertex AI** : Entraînement et versioning des modèles
- **Cloud Run** : API de prédiction serverless
- **Cloud Storage** : Stockage des modèles et artifacts

## 📝 Limitations et Améliorations

### Limitations actuelles

- Pas de gestion des nouveaux produits/agences (cold start)
- Features météo/prix supposées connues à l'avance

### Améliorations possibles

- Ajouter des features de lag pour prédiction court terme
- Implémenter un modèle par catégorie de produit
- Ajouter une validation croisée temporelle
- Intégrer des données externes (météo prévue, calendrier promotionnel)

## 👤 Auteur

Joseph A.

## 📄 License

Ce projet est sous licence MIT.
