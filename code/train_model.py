# code/train_model_week3.py

import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ----------------------------
# Chemins
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, "../data/processed/fake_news_processed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../models")
MODEL_FILE = os.path.join(MODEL_DIR, "rf_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# Création du dossier models si nécessaire
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# Fonction de nettoyage du texte
# ----------------------------
def clean_text(text):
    text = str(text).lower()                      # minuscules
    text = re.sub(r'http\S+', '', text)           # enlever liens
    text = re.sub(r'[^a-z\s]', '', text)         # enlever ponctuation et chiffres
    text = re.sub(r'\s+', ' ', text).strip()     # enlever espaces multiples
    return text

# ----------------------------
# Chargement et nettoyage des données
# ----------------------------
df = pd.read_csv(DATA_FILE)
print(f"Données chargées : {len(df)} lignes")

# Nettoyage du texte
df['processed_text'] = df['processed_text'].apply(clean_text)

# Supprimer les lignes vides
df = df.dropna(subset=['processed_text'])
print(f"Lignes après nettoyage : {len(df)}")

# Séparer X et y
X = df['processed_text']
y = df['label']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {len(X_train)} | Test set: {len(X_test)}")

# ----------------------------
# Vectorisation TF-IDF
# ----------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# Entraînement du modèle RandomForest
# ----------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_vec, y_train)

# ----------------------------
# Évaluation
# ----------------------------
y_pred = clf.predict(X_test_vec)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------
# Sauvegarde modèle et vectorizer
# ----------------------------
joblib.dump(clf, MODEL_FILE)
joblib.dump(vectorizer, VECTORIZER_FILE)
print(f"\nModèle sauvegardé dans : {MODEL_FILE}")
print(f"Vectorizer sauvegardé dans : {VECTORIZER_FILE}")