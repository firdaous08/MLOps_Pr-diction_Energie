---
title: Energy Predictor P5
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---
# Energy Predictor - MLOps Project

Ce projet est une solution complète de **Machine Learning Operations (MLOps)** permettant de prédire la consommation énergétique et les émissions de CO2 de bâtiments de la ville de Seattle.

Il intègre un cycle de vie complet : Entraînement, Packaging, Tests Automatisés, Intégration Continue (CI/CD) et Déploiement.

---

## 🏗 Architecture Technique

Le projet repose sur une architecture moderne et modulaire :

* **API :** [FastAPI](https://fastapi.tiangolo.com/) (Python) pour exposer le modèle.
* **Modèle :** Scikit-Learn (Pipeline pré-entraîné) chargé via Joblib.
* **Base de Données :** PostgreSQL pour stocker l'historique des prédictions (Monitoring).
* **Conteneurisation :** Docker.
* **CI/CD :** GitHub Actions (Tests & Déploiement auto).
* **Hébergement :** Hugging Face Spaces (Docker).

---

## 🚀 Installation et Démarrage (Local)

### Prérequis
* Python 3.9+
* PostgreSQL (Installé localement)
* Git

### 1. Cloner le projet
```bash
git clone [https://github.com/PSEUDO/NOM_DU_REPO.git](https://github.com/PSEUDO/NOM_DU_REPO.git)
cd Projet_5_MLOps