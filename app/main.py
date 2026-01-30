import pandas as pd
import numpy as np
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Import nécessaire pour que joblib reconnaisse la classe personnalisée
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. DEFINITION DE LA CLASSE 
# ==========================================
# classe de l'entraînement,
# sinon joblib.load() plantera car il ne trouvera pas la définition de l'objet

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        SEATTLE_CENTER_LAT = 47.6062
        SEATTLE_CENTER_LON = -122.3321
        
        # Gestion Latitude/Longitude
        X['Latitude'] = X['Latitude'].fillna(SEATTLE_CENTER_LAT)
        X['Longitude'] = X['Longitude'].fillna(SEATTLE_CENTER_LON)
        
        # Haversine
        def haversine_vectorized(lat, lon):
            R = 6371
            phi1, phi2 = np.radians(lat), np.radians(SEATTLE_CENTER_LAT)
            dphi = np.radians(SEATTLE_CENTER_LAT - lat)
            dlambda = np.radians(SEATTLE_CENTER_LON - lon)
            a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return R * c

        X['DistanceFromCenter'] = haversine_vectorized(X['Latitude'], X['Longitude'])

        # Age
        current_year = X['DataYear'] if 'DataYear' in X.columns else 2016
        X['BuildingAge'] = current_year - X['YearBuilt']
        X['BuildingAge'] = X['BuildingAge'].apply(lambda x: max(x, 0))

        # Flags Energie
        for col, new_col in [('SteamUse(kBtu)', 'UsesSteam'), 
                             ('NaturalGas(kBtu)', 'UsesGas'), 
                             ('Electricity(kBtu)', 'UsesElectricity')]:
            if col in X.columns:
                X[new_col] = (X[col].fillna(0) > 0).astype(int)
            else:
                X[new_col] = 0
        
        X['EnergySourceCount'] = X['UsesSteam'] + X['UsesGas'] + X['UsesElectricity']

        # Parking & Ratios
        X['TotalParkingArea'] = X.get('PropertyGFAParking', 0)
        if 'SecondLargestPropertyUseType' in X.columns:
             mask = (X['SecondLargestPropertyUseType'] == 'Parking')
             val_2nd = X.get('SecondLargestPropertyUseTypeGFA', 0).fillna(0)
             if mask.any():
                X.loc[mask, 'TotalParkingArea'] = np.maximum(X.loc[mask, 'TotalParkingArea'], val_2nd.loc[mask])
        
        if 'ThirdLargestPropertyUseType' in X.columns:
             mask = (X['ThirdLargestPropertyUseType'] == 'Parking')
             val_3rd = X.get('ThirdLargestPropertyUseTypeGFA', 0).fillna(0)
             if mask.any():
                X.loc[mask, 'TotalParkingArea'] = np.maximum(X.loc[mask, 'TotalParkingArea'], val_3rd.loc[mask])
        
        denom = X['PropertyGFATotal'].replace(0, np.nan)
        X['ParkingRatio'] = X['TotalParkingArea'] / denom
        X['ParkingRatio'] = X['ParkingRatio'].fillna(0)

        floors = X['NumberofFloors'].replace(0, 1)
        X['SurfacePerFloor'] = X['PropertyGFATotal'] / floors

        # Multi-Use
        if 'SecondLargestPropertyUseType' in X.columns:
            X['IsMultiUse'] = (X['SecondLargestPropertyUseType'].fillna('None') != 'None').astype(int)
        else:
            X['IsMultiUse'] = 0

        return X

# ==========================================
# 2. CHARGEMENT DU MODELE
# ==========================================
app = FastAPI(
    title="API de prédiction de consommation énergétique",
    description="API pour prédire la consommation d'énergie (kBtu) des bâtiments de Seattle.",
    version="1.0"
)

# Chargement unique au démarrage
try:
    pipeline = joblib.load('../models/building_energy_pipeline.joblib')
    print(" Modèle chargé avec succès.")
except Exception as e:
    print(f" Erreur lors du chargement du modèle : {e}")
    pipeline = None

