import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from app.preprocessing import FeatureEngineeringTransformer

# Configuration des chemins 
current_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(current_dir, "../data/2016_Building_Energy_Benchmarking.csv")
MODEL_FILE = os.path.join(current_dir, "../models/building_energy_pipeline.joblib")
RANDOM_STATE = 42

# --------------------------------------------------------------------------------
# 2. FONCTIONS STANDARD
# --------------------------------------------------------------------------------

def load_data(filepath):
    print(f"Chargement {filepath}...")
    return pd.read_csv(filepath)

def clean_data_rows(df):
    """
    Supprime les LIGNES inutiles (Outliers, Non-Conformes).
    """
    print("Filtrage des lignes (Nettoyage)...")
    
    if 'ComplianceStatus' in df.columns:
        df = df[df['ComplianceStatus'] == 'Compliant']
    
    exclude_types = ['Multifamily', 'Residence Hall', 'Senior Care Community']
    df = df[~df['BuildingType'].str.contains('Multifamily', na=False)]
    df = df[~df['PrimaryPropertyType'].isin(exclude_types)]
    
    if 'SiteEUI(kBtu/sf)' in df.columns:
        df = df[(df['SiteEUI(kBtu/sf)'] > 0) & (df['SiteEUI(kBtu/sf)'] < 500)]
    
    return df.copy()

# --------------------------------------------------------------------------------
# 3. PIPELINE & ENTRAÎNEMENT
# --------------------------------------------------------------------------------

def train_pipeline(df):
    
    # 1. Séparation X / y
    target_col = 'SiteEnergyUse(kBtu)'
    
    # On prépare X
    X = df.drop(columns=[target_col, 'SiteEUI(kBtu/sf)'], errors='ignore')
    
    # On garde les infos d'énergie juste pour créer les flags 0/1 dans le transformer
    # (Le transformer s'occupe de gérer ça sans fuite de données numérique)
    
    # Log Transformation de la cible
    y = np.log1p(df[target_col])
    
    # 2. Définition des colonnes générées par le Transformer
    numeric_features = [
        'PropertyGFATotal', 'PropertyGFAParking', 'NumberofBuildings', 'NumberofFloors', 
        'SurfacePerFloor', 'ParkingRatio', 'BuildingAge', 
        'Latitude', 'Longitude', 'DistanceFromCenter', 
        'IsMultiUse', 'EnergySourceCount', 'UsesSteam', 'UsesGas', 'UsesElectricity'
    ]
    categorical_features = ['Neighborhood', 'BuildingType', 'PrimaryPropertyType']

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 4. Construction du Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        verbose_feature_names_out=False
    )

    full_pipeline = Pipeline([
        ('feature_engineering', FeatureEngineeringTransformer()), # Vient de preprocessing.py
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
    ])

    # 5. Entraînement
    print("Entraînement du Pipeline complet...")
    full_pipeline.fit(X_train, y_train)

    # 6. Eval
    print("Evaluation...")
    y_pred = full_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE (Log Scale): {rmse:.4f}")

    # 7. Sauvegarde
    # Création du dossier models s'il n'existe pas
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(full_pipeline, MODEL_FILE)
    print(f" Modèle sauvegardé : {MODEL_FILE}")

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        df = load_data(INPUT_FILE)
        df = clean_data_rows(df)
        train_pipeline(df)
    else:
        print(f"ERREUR : Fichier introuvable : {INPUT_FILE}")