# 🔍 TruthLens — Fake News Detection

> Projet final — Cours Python pour Data Science  
> **Wala Mhatli**
> **Omar ghannem**
> **Oumaima ghannem**

---

## 📌 Description

TruthLens est une application complète de détection de fake news basée sur le machine learning. Elle analyse des articles de presse et prédit s'ils sont **réels ou fabriqués**, avec un score de crédibilité et une explication des mots clés qui ont influencé la décision.

---

## 🏗️ Architecture

```
fake_news_project/
├── api/                  # FastAPI backend
│   └── main.py
├── src/                  # Code ML
│   ├── preprocess.py     # Nettoyage & feature engineering
│   ├── predict.py        # Classe de prédiction
│   └── train.py          # Pipeline d'entraînement
├── frontend/             # Interface React
├── notebooks/            # Analyses & explainability
│   ├── week3_modeling_mlflow.ipynb
│   └── week7_explainability.ipynb
├── data/                 # Modèle & artefacts (non versionnés)
├── tests/                # Tests unitaires & intégration
├── Dockerfile.backend
├── Dockerfile.frontend
└── docker-compose.yml
```

---

## 🤖 Modèle

- **Algorithme** : LightGBM
- **Features** : TF-IDF (50 000 tokens) + 7 features artisanales
- **Dataset** : ~44 000 articles (Fake & Real News Dataset)
- **Performance** : ~99% accuracy sur le jeu de test
- **Tracking** : MLflow

### Features artisanales
| Feature | Description |
|---|---|
| `title_word_count` | Nombre de mots dans le titre |
| `text_word_count` | Nombre de mots dans le texte |
| `exclamation_count` | Nombre de points d'exclamation |
| `question_count` | Nombre de points d'interrogation |
| `uppercase_ratio` | Ratio de lettres en majuscules |
| `suspicious_keyword_count` | Mots sensationnalistes détectés |
| `avg_word_length` | Longueur moyenne des mots |

---

## 🚀 Lancement rapide

### Prérequis
- Docker & Docker Compose
- Les fichiers modèle dans `data/` (`best_model.pkl`, `tfidf_vectorizer.pkl`, `feature_cols.json`)

### Démarrage
```bash
docker compose up -d
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| MLflow | http://localhost:5000 |
| Docs API | http://localhost:8000/docs |

---

## 🔌 API

### `POST /predict`
```json
{
  "title": "Federal Reserve raises interest rates",
  "text": "WASHINGTON (Reuters) - The Federal Reserve..."
}
```

**Réponse :**
```json
{
  "label": "REAL",
  "confidence": 0.9998,
  "prob_real": 0.9998,
  "prob_fake": 0.0002,
  "features": { ... }
}
```

### `POST /explain`
Retourne les mots qui ont le plus influencé la prédiction (LIME).

### `POST /predict/batch`
Analyse plusieurs articles en une seule requête.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

34 tests unitaires et d'intégration couvrant le preprocessing et l'API.

---

## 📊 Explainability

Le projet inclut un notebook complet (Week 7) avec :
- **SHAP** : importance globale des features sur l'ensemble du dataset
- **LIME** : explication locale article par article
- Visualisations sauvegardées dans `data/shap_summary.png`, `data/lime_real.png`, `data/lime_fake.png`

---

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| ML | LightGBM, scikit-learn |
| Explainability | SHAP, LIME |
| Backend | FastAPI, Python 3.11 |
| Frontend | React, Tailwind CSS |
| Tracking | MLflow |
| Déploiement | Docker, Docker Compose |
| Tests | pytest |

---

*Projet réalisé dans le cadre du cours Python pour Data Science — 2026*
