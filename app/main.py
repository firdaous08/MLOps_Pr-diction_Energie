import pandas as pd
import numpy as np
import joblib
import uvicorn
import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.preprocessing import FeatureEngineeringTransformer
from app.database import get_db, PredictionLog 

# ==========================================
# 1. CONFIG ET MODELE
# ==========================================
app = FastAPI(
    title="API de prédiction de la consomation énergétique des bâtiments non résidentiels",
    description="API MLOps avec Logging en Base de Données (PostgreSQL).",
    version="1.0"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
# Ajustement du chemin pour pointer vers le bon dossier models
model_path = os.path.join(current_dir, "../models/building_energy_pipeline.joblib")

pipeline = None
try:
    pipeline = joblib.load(model_path)
    print(" Modèle chargé avec succès !")
except Exception as e:
    print(f"ERREUR : Impossible de charger le modèle : {e}")

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

@app.post("/predict")
def predict_energy(data: BuildingInput, db: Session = Depends(get_db)):
    # Vérification que le modèle est bien là
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé côté serveur")

    # A. Prédiction (Partie critique qui doit toujours marcher)
    try:
        # Conversion Pydantic -> Dict -> DataFrame
        # Note : Si tu as une vieille version de Pydantic (<2.0), utilise .dict() au lieu de .model_dump()
        input_data_dict = data.dict() if hasattr(data, 'dict') else data.model_dump()
        
        df_input = pd.DataFrame([input_data_dict])
        
        # Prédiction
        prediction_log = pipeline.predict(df_input)
        prediction_kbtu = np.expm1(prediction_log[0])
        result = round(float(prediction_kbtu), 2)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {str(e)}")

    # B. Enregistrement en Base de Données (Optionnel / Fail-Safe)
    log_id = None
    warning_msg = None

    try:
        new_log = PredictionLog(
            input_data=input_data_dict,
            predicted_value=result,
            model_version="1.0"
        )
        db.add(new_log)
        db.commit()
        db.refresh(new_log)
        log_id = new_log.id
        print(" Données sauvegardées en BDD.")
        
    except Exception as e:
        # On attrape l'erreur sans faire planter l'API
        print(f" ATTENTION : Impossible de sauvegarder en BDD : {e}")
        warning_msg = "Prédiction réussie, mais sauvegarde impossible (BDD non connectée)."

    # C. Retourner la réponse
    return {
        "prediction_kbtu": result,
        "log_id": log_id,
        "status": "success",
        "warning": warning_msg
    }