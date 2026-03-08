import sys, pickle, json, re
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
sys.path.insert(0, '.')
from src.preprocess import clean_text, extract_handcrafted_features, FEATURE_COLS

def remove_dateline(text):
    """Supprime le préfixe 'CITY (Reuters) - ' des vrais articles"""
    return re.sub(r'^[A-Z\s]+\([^)]+\)\s*-\s*', '', str(text))

print("Chargement...")
true_df = pd.read_csv('data/True.csv')
fake_df = pd.read_csv('data/Fake.csv')
true_df['label'] = 1
fake_df['label'] = 0

df = pd.concat([true_df, fake_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# ⚠️ Supprime le dateline Reuters (source du leakage)
df['text'] = df['text'].apply(remove_dateline)

df = extract_handcrafted_features(df, include_subject=False)
feature_cols = list(FEATURE_COLS)
X_meta = csr_matrix(df[feature_cols].values.astype(np.float32))

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True, min_df=5, max_df=0.9)
X_tfidf = tfidf.fit_transform(df['combined'])

X = hstack([X_tfidf, X_meta]).tocsr()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"X_train: {X_train.shape}")

print("Entrainement...")
lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, n_jobs=-1, verbose=-1)
lgbm.fit(X_train, y_train)

print("Evaluation...")
y_pred = lgbm.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

print("Sauvegarde...")
pickle.dump(lgbm, open('data/best_model.pkl','wb'))
pickle.dump(tfidf, open('data/tfidf_vectorizer.pkl','wb'))
json.dump(feature_cols, open('data/feature_cols.json','w'))
print("✅ Tout sauvegardé !")