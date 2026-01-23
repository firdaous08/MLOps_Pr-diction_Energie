# Projet 5 : Déploiement de Modèle ML (Classification)

Ce projet expose une API permettant de classifier des textes automatiquement via un modèle de Machine Learning.

## Structure du projet
- `app/` : Code source de l'API (FastAPI)
- `model/` : Fichiers du modèle entraîné (.joblib)
- `tests/` : Tests unitaires

## Installation
1. Cloner le dépôt.
2. Créer un environnement virtuel : `python3 -m venv .venv`
3. Installer les dépendances : `pip install -r requirements.txt`

## Lancement
Lancer l'API : `uvicorn app.main:app --reload`