import pandas as pd
import numpy as np
import joblib
import uvicorn
import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.preprocessing import FeatureEngineeringTransformer
from app.database import get_db, PredictionLog # imports BDD

# ==========================================
# 1. CONFIG ET MODELE
# ==========================================
app = FastAPI(
    title="API de prédiction de la consomation énergétique des bâtiments non résidentiels ",
    description="API MLOps avec Logging en Base de Données (PostgreSQL).",
    version="1.0"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../models/building_energy_pipeline.joblib")

pipeline = None
try:
    pipeline = joblib.load(model_path)
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f" ERREUR : Impossible de charger le modèle : {e}")

# ============================================================
# 2. INPUT DATA
# ============================================================
class BuildingInput(BaseModel):
    BuildingType: str
    PrimaryPropertyType: str
    Neighborhood: str
    YearBuilt: int
    NumberofBuildings: float
    NumberofFloors: float
    PropertyGFATotal: float
    PropertyGFAParking: float = 0.0 
    Latitude: float
    Longitude: float
    UsesSteam: int = 0
    UsesGas: int = 0
    UsesElectricity: int = 1
    IsMultiUse: int = 0

# ============================================================
# 3. ROUTES
# ============================================================
@app.get("/")
def home():
    return {"message": "API de Prédiction Energétique en ligne !"}

# On ajoute le paramètre 'db' pour avoir accès à la base
@app.post("/predict")
def predict_energy(data: BuildingInput, db: Session = Depends(get_db)):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé côté serveur")

    try:
        # A. Prédiction
        df_input = pd.DataFrame([data.model_dump()])
        prediction_log = pipeline.predict(df_input)
        prediction_kbtu = np.expm1(prediction_log[0])
        result = round(float(prediction_kbtu), 2)

        # B. Enregistrement en Base de Données (Logging)
        new_log = PredictionLog(
            input_data=data.model_dump(),  # On sauvegarde tout le JSON d'entrée
            predicted_value=result,        # On sauvegarde la prédiction
            model_version="1.0"
        )
        db.add(new_log)  # On ajoute la ligne
        db.commit()      # On valide la transaction
        db.refresh(new_log) # On récupère l'ID créé par la BDD

        return {
            "prediction_kbtu": result,
            "log_id": new_log.id,  # On prouve à l'utilisateur que c'est enregistré
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur : {str(e)}")