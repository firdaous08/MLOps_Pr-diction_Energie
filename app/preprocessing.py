# Fichier: app/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# On définit les constantes ici pour qu'elles soient accessibles
SEATTLE_CENTER_LAT = 47.6062
SEATTLE_CENTER_LON = -122.3321

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # --- A. Distance (Haversine) ---
        def haversine_vectorized(lat, lon):
            R = 6371
            phi1, phi2 = np.radians(lat), np.radians(SEATTLE_CENTER_LAT)
            dphi = np.radians(SEATTLE_CENTER_LAT - lat)
            dlambda = np.radians(SEATTLE_CENTER_LON - lon)
            a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return R * c

        X['Latitude'] = X['Latitude'].fillna(SEATTLE_CENTER_LAT)
        X['Longitude'] = X['Longitude'].fillna(SEATTLE_CENTER_LON)
        X['DistanceFromCenter'] = haversine_vectorized(X['Latitude'], X['Longitude'])

        # --- B. Age ---
        current_year = 2016 
        # Si la colonne DataYear existe on l'utilise, sinon 2016
        years = X['DataYear'] if 'DataYear' in X.columns else current_year
        
        if 'YearBuilt' in X.columns:
            X['BuildingAge'] = years - X['YearBuilt']
            X['BuildingAge'] = X['BuildingAge'].fillna(0).apply(lambda x: max(x, 0))
        else:
            X['BuildingAge'] = 0

        # --- C. Flags Sources ---
        # Mapping des noms bruts (CSV) vers tes noms internes
        source_map = {
            'SteamUse(kBtu)': 'UsesSteam',
            'NaturalGas(kBtu)': 'UsesGas',
            'Electricity(kBtu)': 'UsesElectricity'
        }
        
        # Pour les colonnes qui existent déjà dans X (si input API)
        # OU pour les colonnes du CSV d'origine
        for col_csv, col_api in source_map.items():
            if col_api in X.columns:
                # Si déjà présent (cas API), on s'assure que c'est 0 ou 1
                 X[col_api] = X[col_api].fillna(0).astype(int)
            elif col_csv in X.columns:
                # Si format CSV, on transforme en binaire
                X[col_api] = (X[col_csv].fillna(0) > 0).astype(int)
            else:
                # Sinon par défaut 0
                X[col_api] = 0
        
        X['EnergySourceCount'] = X['UsesSteam'] + X['UsesGas'] + X['UsesElectricity']

        # --- D. Parking & Ratios ---
        X['TotalParkingArea'] = X.get('PropertyGFAParking', 0)
        
        # Logique complexe Parking (simplifiée pour l'API si les colonnes manquent)
        if 'SecondLargestPropertyUseType' in X.columns:
             mask = (X['SecondLargestPropertyUseType'] == 'Parking')
             val_2nd = X.get('SecondLargestPropertyUseTypeGFA', 0).fillna(0)
             if mask.any():
                X.loc[mask, 'TotalParkingArea'] = np.maximum(X.loc[mask, 'TotalParkingArea'], val_2nd.loc[mask])

        denom = X['PropertyGFATotal'].replace(0, np.nan)
        X['ParkingRatio'] = X['TotalParkingArea'] / denom
        X['ParkingRatio'] = X['ParkingRatio'].fillna(0)

        floors = X['NumberofFloors'].replace(0, 1)
        X['SurfacePerFloor'] = X['PropertyGFATotal'] / floors

        # --- E. IsMultiUse ---
        if 'SecondLargestPropertyUseType' in X.columns:
            X['IsMultiUse'] = (X['SecondLargestPropertyUseType'].fillna('None') != 'None').astype(int)
        elif 'IsMultiUse' not in X.columns:
             X['IsMultiUse'] = 0

        return X