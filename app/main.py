import pandas as pd
import numpy as np
import joblib
import uvicorn
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# --- IMPORT CRUCIAL : C'est ici que la magie opère ---
from app.preprocessing import FeatureEngineeringTransformer

# ==========================================
# 2. CHARGEMENT DU MODÈLE
# ==========================================
app = FastAPI(
    title="Building Energy Prediction API",
    description="API pour prédire la consommation d'énergie (kBtu) des bâtiments de Seattle.",
    version="1.0"
)

# 1. On trouve où est le fichier main.py sur le disque
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. On construit le chemin vers le modèle
model_path = os.path.join(current_dir, "../models/building_energy_pipeline.joblib")

print(f"🔍 Je cherche le modèle ici : {model_path}")

pipeline = None

try:
    pipeline = joblib.load(model_path)
    print("✅ Modèle chargé avec succès !")
except Exception as e:
    print(f"❌ ERREUR : Impossible de charger le modèle.")
    print(f"❌ Détail : {e}")

# ============================================================
# 3. DÉFINITION DES DONNÉES ENTRANTES
# ============================================================
class BuildingInput(BaseModel):
    BuildingType: str          # ex: "Office"
    PrimaryPropertyType: str   # ex: "Small Office"
    Neighborhood: str          # ex: "DOWNTOWN"
    YearBuilt: int             # ex: 1990
    NumberofBuildings: float   # ex: 1
    NumberofFloors: float      # ex: 3
    PropertyGFATotal: float    # ex: 25000
    PropertyGFAParking: float = 0.0 
    Latitude: float            # ex: 47.61
    Longitude: float           # ex: -122.33
    UsesSteam: int = 0
    UsesGas: int = 0
    UsesElectricity: int = 1
    IsMultiUse: int = 0

# ============================================================
# 4. ROUTES
# ============================================================
@app.get("/")
def home():
    return {"message": "API de Prédiction Énergétique en ligne !"}

@app.post("/predict")
def predict_energy(data: BuildingInput):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé côté serveur")

    try:
        # 1. Création DataFrame
        df_input = pd.DataFrame([data.model_dump()])
        # 2. Prédiction (Le pipeline va appeler preprocessing.py automatiquement)
        prediction_log = pipeline.predict(df_input)

        # 3. Inverse Log
        prediction_kbtu = np.expm1(prediction_log[0])

        return {
            "prediction_kbtu": round(float(prediction_kbtu), 2),
            "log_value": round(float(prediction_log[0]), 3),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {str(e)}")