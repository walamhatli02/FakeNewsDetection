# 📋 GUIDE COMPLET – Fake News Detection
## Python for Data Science 2 – Ce que tu dois faire exactement

---

## 🗂️ STRUCTURE DU PROJET (déjà créée pour toi)

```
fake_news_project/
├── notebooks/
│   ├── week1_eda.ipynb              ← Semaine 1 : EDA
│   ├── week2_preprocessing.ipynb   ← Semaine 2 : Preprocessing
│   └── week3_modeling_mlflow.ipynb ← Semaine 3 : Modèles + MLflow
├── src/
│   ├── preprocess.py               ← Pipeline de preprocessing
│   ├── train.py                    ← Script d'entraînement CLI
│   ├── predict.py                  ← Classe de prédiction
│   └── evaluate.py                 ← Métriques + graphes
├── api/
│   └── main.py                     ← API FastAPI (Semaine 4)
├── frontend/
│   ├── src/App.jsx                 ← Interface React (Semaine 5)
│   └── src/index.js
├── tests/
│   ├── test_api.py                 ← Tests de l'API
│   └── test_preprocess.py         ← Tests du preprocessing
├── Dockerfile.backend              ← Semaine 6
├── Dockerfile.frontend             ← Semaine 6
├── docker-compose.yml              ← Semaine 6
└── requirements.txt
```

---

## 📦 ÉTAPE 0 — Installation (à faire UNE SEULE FOIS)

### 0.1 — Cloner / créer le dossier
```bash
# Extraire le zip téléchargé, puis aller dans le dossier :
cd fake_news_project
```

### 0.2 — Créer un environnement virtuel Python
```bash
python -m venv venv

# Activer l'environnement :
# Windows :
venv\Scripts\activate
# Mac/Linux :
source venv/bin/activate
```

### 0.3 — Installer les dépendances
```bash
pip install -r requirements.txt
```

### 0.4 — Télécharger le dataset
1. Va sur : https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Crée un compte Kaggle gratuit si tu n'en as pas
3. Clique sur **Download** (bouton en haut à droite)
4. Extrait le ZIP → tu obtiens **True.csv** et **Fake.csv**
5. Place ces deux fichiers dans le dossier `data/` du projet

```
fake_news_project/data/True.csv   ✅
fake_news_project/data/Fake.csv   ✅
```

---

## 📊 SEMAINE 1 — EDA (Exploratory Data Analysis)

### Ce que tu dois faire :
1. Ouvre Jupyter Notebook :
   ```bash
   jupyter notebook
   ```
2. Navigue vers `notebooks/week1_eda.ipynb`
3. **Exécute toutes les cellules** dans l'ordre (Shift+Enter ou Run All)

### Ce que le notebook produit :
- Statistiques descriptives du dataset (44 000 articles)
- Graphiques de distribution (sauvegardés dans `data/`)
- Fichier `data/raw_data.csv` pour la semaine 2

### ✅ Résultat attendu :
```
Total samples  : 44,898
Real news      : 21,417
Fake news      : 23,481
```

---

## 🔧 SEMAINE 2 — Preprocessing & Feature Engineering

### Ce que tu dois faire :
1. Dans Jupyter, ouvre `notebooks/week2_preprocessing.ipynb`
2. Exécute toutes les cellules dans l'ordre

### Ce que le notebook fait :
- **Nettoyage du texte** : supprime URLs, HTML, ponctuation, normalise
- **TF-IDF** : transforme le texte en 50 000 features numériques
- **Features artisanaux** : ratio majuscules, longueur, ponctuation...
- **Train/Test split** : 80% train / 20% test (stratifié)

### ✅ Fichiers produits dans `data/` :
```
X_train.npz          ← Matrice features d'entraînement
X_test.npz           ← Matrice features de test
y_train.csv          ← Labels d'entraînement
y_test.csv           ← Labels de test
tfidf_vectorizer.pkl ← Vectoriseur TF-IDF sauvegardé
feature_cols.json    ← Noms des features
```

---

## 🤖 SEMAINE 3 — Modélisation + MLflow

### Étape 1 : Lancer le serveur MLflow (dans un terminal séparé)
```bash
mlflow ui --port 5000
```
> Garde ce terminal ouvert. MLflow sera disponible sur http://localhost:5000

### Étape 2 : Exécuter le notebook de modélisation
1. Dans Jupyter, ouvre `notebooks/week3_modeling_mlflow.ipynb`
2. Exécute toutes les cellules

### Ce que le notebook fait :
- **Modèle 1** : Logistic Regression (baseline)
- **Modèle 2** : XGBoost
- **Modèle 3** : LightGBM (meilleur résultat)
- Track tous les paramètres et métriques dans MLflow
- Génère la matrice de confusion

### ✅ Résultats attendus :
| Modèle | Accuracy | ROC-AUC |
|--------|----------|---------|
| Logistic Regression | ~98% | ~0.998 |
| XGBoost | ~99% | ~0.999 |
| **LightGBM** ✅ | **~99%** | **~0.999** |

### ✅ Fichiers produits :
```
data/best_model.pkl       ← Modèle LightGBM sauvegardé
data/confusion_matrix.png ← Graphe de la matrice de confusion
```

### Voir les expériences MLflow :
- Ouvre http://localhost:5000 dans ton navigateur
- Tu verras les 3 runs avec tous les paramètres et métriques
- Tu peux comparer les modèles visuellement

**Alternative CLI (sans notebook) :**
```bash
python -m src.train --data_dir data/ --output_dir data/
```

---

## 🔌 SEMAINE 4 — API FastAPI

### Étape 1 : Lancer l'API
```bash
# Depuis la racine du projet :
uvicorn api.main:app --reload --port 8000
```

### Étape 2 : Tester l'API
1. **Interface interactive** : http://localhost:8000/docs
   - Clique sur `/predict` → "Try it out" → entre un titre et un texte

2. **Test avec curl** :
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Federal Reserve raises interest rates",
    "text": "The Federal Reserve raised its benchmark interest rate by a quarter point on Wednesday. Fed Chair Jerome Powell said the decision was unanimous among voting members."
  }'
```

3. **Résultat attendu** :
```json
{
  "label": "REAL",
  "label_id": 1,
  "confidence": 0.9874,
  "real_probability": 0.9874,
  "fake_probability": 0.0126
}
```

### Étape 3 : Exécuter les tests automatiques
```bash
pip install pytest httpx
pytest tests/test_api.py -v
```

### Endpoints disponibles :
| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Info de base |
| `/health` | GET | Statut de l'API |
| `/docs` | GET | Documentation Swagger |
| `/examples` | GET | Exemples de news |
| `/predict` | POST | Prédire 1 article |
| `/predict/batch` | POST | Prédire jusqu'à 50 articles |

---

## 🎨 SEMAINE 5 — Frontend React

### Étape 1 : Installer Node.js
- Télécharge Node.js (version LTS) : https://nodejs.org
- Vérifie : `node --version` (doit afficher v18 ou plus)

### Étape 2 : Installer les dépendances
```bash
cd frontend
npm install
```

### Étape 3 : S'assurer que l'API est déjà lancée
```bash
# Dans un terminal séparé :
uvicorn api.main:app --reload --port 8000
```

### Étape 4 : Lancer l'interface React
```bash
# Depuis le dossier frontend/ :
npm start
```

L'interface s'ouvre automatiquement sur http://localhost:3000

### Ce que l'interface permet :
- Saisir un titre et un texte d'article
- Choisir le sujet (optionnel)
- Tester avec des exemples pré-remplis
- Voir le résultat avec barre de probabilité animée

---

## 🐳 SEMAINE 6 — Containerisation Docker

### Étape 1 : Installer Docker
- Télécharge Docker Desktop : https://www.docker.com/products/docker-desktop/
- Vérifie : `docker --version`

### Étape 2 : S'assurer que les fichiers modèles existent
```bash
# Ces fichiers doivent exister (générés par les notebooks) :
ls data/best_model.pkl
ls data/tfidf_vectorizer.pkl
ls data/feature_cols.json
```

### Étape 3 : Lancer tout le projet avec Docker
```bash
# Depuis la racine du projet :
docker-compose up --build
```

> ⚠️ La première fois prend 5-10 minutes (téléchargement des images)

### Services disponibles après docker-compose up :
| Service | URL | Description |
|---------|-----|-------------|
| API Backend | http://localhost:8000 | FastAPI |
| Interface Web | http://localhost:3000 | React |
| MLflow | http://localhost:5000 | Tracking |

### Commandes Docker utiles :
```bash
# Lancer en arrière-plan
docker-compose up -d

# Voir les logs
docker-compose logs -f

# Arrêter tous les services
docker-compose down

# Reconstruire après modification
docker-compose up --build
```

---

## ✅ CHECKLIST FINALE

Avant de rendre ton projet, vérifie que tu as :

### Semaine 1 — EDA
- [ ] Notebook `week1_eda.ipynb` exécuté sans erreur
- [ ] Au moins 4 visualisations produites et sauvegardées dans `data/`
- [ ] Fichier `data/raw_data.csv` généré

### Semaine 2 — Preprocessing
- [ ] Notebook `week2_preprocessing.ipynb` exécuté sans erreur
- [ ] Fichiers `X_train.npz`, `X_test.npz`, `y_train.csv`, `y_test.csv` présents
- [ ] `tfidf_vectorizer.pkl` et `feature_cols.json` présents

### Semaine 3 — Modélisation
- [ ] Notebook `week3_modeling_mlflow.ipynb` exécuté sans erreur
- [ ] 3 modèles entraînés et comparés
- [ ] Résultats visibles dans MLflow (http://localhost:5000)
- [ ] `best_model.pkl` présent dans `data/`
- [ ] Accuracy ≥ 95% sur le test set

### Semaine 4 — API
- [ ] API se lance avec `uvicorn api.main:app --reload`
- [ ] `/predict` retourne un JSON bien formé
- [ ] Tests passent : `pytest tests/ -v`
- [ ] Documentation Swagger accessible sur `/docs`

### Semaine 5 — Frontend
- [ ] `npm install` réussit sans erreur
- [ ] `npm start` lance l'interface sur localhost:3000
- [ ] L'interface communique avec l'API et affiche les résultats

### Semaine 6 — Docker
- [ ] `docker-compose up --build` démarre les 3 services
- [ ] API accessible sur port 8000 dans Docker
- [ ] Frontend accessible sur port 3000 dans Docker

---

## 🚨 PROBLÈMES FRÉQUENTS

### "ModuleNotFoundError: No module named 'lightgbm'"
```bash
pip install lightgbm xgboost
```

### "FileNotFoundError: best_model.pkl"
→ Tu dois d'abord exécuter le notebook de la Semaine 3 pour générer le modèle.

### L'API dit "503 Model not loaded"
→ Exécute d'abord les notebooks S1 → S2 → S3, puis relance l'API.

### "CORS Error" dans React
→ Vérifie que l'API tourne sur le port 8000 et que CORS est activé (c'est déjà le cas dans le code).

### Docker ne trouve pas les fichiers modèles
```bash
# S'assurer que ces fichiers existent AVANT le docker-compose up :
ls data/best_model.pkl
ls data/tfidf_vectorizer.pkl
```

### MLflow ne démarre pas sur le port 5000
```bash
mlflow ui --port 5001  # Essaie un autre port
```

---

## 📌 ORDRE D'EXÉCUTION COMPLET (résumé)

```
1. pip install -r requirements.txt
2. Télécharger True.csv + Fake.csv → dossier data/
3. jupyter notebook → week1_eda.ipynb          [S1]
4. jupyter notebook → week2_preprocessing.ipynb [S2]
5. mlflow ui --port 5000  (terminal séparé)
6. jupyter notebook → week3_modeling_mlflow.ipynb [S3]
7. uvicorn api.main:app --reload --port 8000    [S4]
8. cd frontend && npm install && npm start       [S5]
9. docker-compose up --build                    [S6]
```

---

*Projet réalisé pour le cours Python for Data Science 2 – Haythem Ghazouani*
