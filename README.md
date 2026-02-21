1/ Projet Détection de Fake News

* Description
Ce projet vise à détecter les fake news en utilisant un classifieur Random Forest entraîné sur des features TF-IDF.  
Le notebook associé montre l'analyse exploratoire des données, la visualisation, l'entraînement du modèle et son évaluation.



2/ Structure du projet


FakeNewsDetection/
├── data/
│ ├── raw/ # Données brutes
│ └── processed/ # Données pré-traitées (ignorées par Git)
├── code/
│ ├── train_model.py # Script d'entraînement du modèle
│ └── preprocessing.py # Script de prétraitement des données
├── models/
│ ├── rf_model.pkl # Modèle Random Forest sauvegardé
│ └── tfidf_vectorizer.pkl # Vectorizer TF-IDF
├── notebooks/
│ └── FakeNews_Detection.ipynb # Notebook complet EDA + modèle
├── .gitignore
├── requirements.txt
└── README.md




3/ Installation

3.1) Cloner le repo :
-bash
git clone https://github.com/walamhatli02/FakeNewsDetection.git
cd FakeNewsDetection

3.2) Créer un environnement virtuel et activer :

python -m venv venv
-Windows
venv\Scripts\activate
-Mac/Linux
source venv/bin/activate

3.3) Installer les dépendances :

pip install -r requirements.txt
4/ Utilisation
1. Prétraitement des données:

python code/preprocessing.py

2. Entraînement du modèle: 

python code/train_model.py

3. Exploration et visualisation

 Ouvrir le notebook : notebooks/FakeNews_Detection.ipynb

Contient :

* Analyse exploratoire des données (EDA)

* Répartition des classes et longueur des textes

* WordClouds pour Fake et Real News

* Évaluation du modèle

* Prédictions sur des exemples
4/ Résultats

-Accuracy du modèle : environ 99,78% sur l'ensemble de test

-Matrice de confusion : montre la performance sur chaque classe

-WordClouds : illustrent les mots les plus fréquents pour Fake vs Real News

****Notes

Les fichiers volumineux (ex. données traitées .csv) sont ignorés par Git grâce à .gitignore.

Le notebook contient des cellules Markdown expliquant chaque étape (mini rapport scientifique).