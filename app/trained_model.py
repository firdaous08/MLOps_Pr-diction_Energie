import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configuration
INPUT_FILE = '../data/2016_Building_Energy_Benchmarking.csv'
MODEL_FILE = '../models/building_energy_pipeline.joblib'
RANDOM_STATE = 42

# Coordonnées Seattle (Constantes globales)
SEATTLE_CENTER_LAT = 47.6062
SEATTLE_CENTER_LON = -122.3321

# --------------------------------------------------------------------------------
# 1. CUSTOM TRANSFORMER (Le cerveau du Feature Engineering)
# --------------------------------------------------------------------------------
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Cette classe transforme les données brutes en features utilisables par le modèle.
    Elle est sauvegardée AVEC le pipeline. L'API n'aura plus besoin de refaire les calculs.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # Rien à apprendre (pas de moyenne/écart-type à retenir)

    def transform(self, X):
        # On travaille sur une copie pour ne pas modifier l'original
        X = X.copy()
        
        # --- A. Calcul Distance Haversine ---
        # On définit la fonction ici ou on utilise vectorisation numpy
        def haversine_vectorized(lat, lon):
            R = 6371
            phi1, phi2 = np.radians(lat), np.radians(SEATTLE_CENTER_LAT)
            dphi = np.radians(SEATTLE_CENTER_LAT - lat)
            dlambda = np.radians(SEATTLE_CENTER_LON - lon)
            a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return R * c

        # Gestion des NaN pour lat/lon (valeur par défaut pour éviter crash)
        X['Latitude'] = X['Latitude'].fillna(SEATTLE_CENTER_LAT)
        X['Longitude'] = X['Longitude'].fillna(SEATTLE_CENTER_LON)
        X['DistanceFromCenter'] = haversine_vectorized(X['Latitude'], X['Longitude'])

        # --- B. Age du bâtiment ---
        # On suppose que l'année courante est celle des données (2016) ou celle de l'input
        current_year = 2016 
        if 'DataYear' in X.columns:
             # Si DataYear existe (entraînement), on l'utilise, sinon fallback
             years = X['DataYear']
        else:
             years = current_year
        
        X['BuildingAge'] = years - X['YearBuilt']
        X['BuildingAge'] = X['BuildingAge'].apply(lambda x: max(x, 0)) # Pas d'âge négatif

        # --- C. Flags Sources Énergie ---
        # Note : Pour l'API, il faudra s'assurer que ces colonnes existent ou sont créées
        sources = ['SteamUse(kBtu)', 'NaturalGas(kBtu)', 'Electricity(kBtu)']
        targets = ['UsesSteam', 'UsesGas', 'UsesElectricity']
        
        for src, tgt in zip(sources, targets):
            if src in X.columns:
                X[tgt] = (X[src].fillna(0) > 0).astype(int)
            else:
                X[tgt] = 0 # Par défaut si info manquante
        
        X['EnergySourceCount'] = X['UsesSteam'] + X['UsesGas'] + X['UsesElectricity']

        # --- D. Parking & Ratios ---
        # Consolidation surface parking
        X['TotalParkingArea'] = X.get('PropertyGFAParking', 0)
        
        # Logique parking caché (Si colonnes dispos)
        if 'SecondLargestPropertyUseType' in X.columns:
             mask = (X['SecondLargestPropertyUseType'] == 'Parking')
             # On utilise fillna(0) pour sécuriser les calculs
             val_2nd = X.get('SecondLargestPropertyUseTypeGFA', 0).fillna(0)
             # Mise à jour conditionnelle vectorisée
             X.loc[mask, 'TotalParkingArea'] = np.maximum(X.loc[mask, 'TotalParkingArea'], val_2nd.loc[mask])

        if 'ThirdLargestPropertyUseType' in X.columns:
             mask = (X['ThirdLargestPropertyUseType'] == 'Parking')
             val_3rd = X.get('ThirdLargestPropertyUseTypeGFA', 0).fillna(0)
             X.loc[mask, 'TotalParkingArea'] = np.maximum(X.loc[mask, 'TotalParkingArea'], val_3rd.loc[mask])
        
        # Ratio Parking
        denom = X['PropertyGFATotal'].replace(0, np.nan)
        X['ParkingRatio'] = X['TotalParkingArea'] / denom
        X['ParkingRatio'] = X['ParkingRatio'].fillna(0)

        # Surface Per Floor
        floors = X['NumberofFloors'].replace(0, 1) # Eviter div/0
        X['SurfacePerFloor'] = X['PropertyGFATotal'] / floors

        # --- E. IsMultiUse ---
        if 'SecondLargestPropertyUseType' in X.columns:
            # On traite les NaN comme des chaînes pour la comparaison
            X['IsMultiUse'] = (X['SecondLargestPropertyUseType'].fillna('None') != 'None').astype(int)
        else:
            X['IsMultiUse'] = 0

        # On retourne le DataFrame enrichi
        return X

# --------------------------------------------------------------------------------
# 2. FONCTIONS STANDARD (Chargement, Nettoyage Lignes)
# --------------------------------------------------------------------------------

def load_data(filepath):
    print(f"Chargement {filepath}...")
    return pd.read_csv(filepath)

def clean_data_rows(df):
    """
    Supprime les LIGNES inutiles (Outliers, Non-Conformes).
    Ceci reste hors du pipeline car le pipeline transforme des colonnes, il ne supprime pas de lignes en prod.
    """
    print("Filtrage des lignes (Nettoyage)...")
    
    # Filtres de base
    if 'ComplianceStatus' in df.columns:
        df = df[df['ComplianceStatus'] == 'Compliant']
    
    # Filtres types résidentiels
    exclude_types = ['Multifamily', 'Residence Hall', 'Senior Care Community']
    # Simplification du filtre pour l'exemple (à adapter selon ta logique stricte)
    df = df[~df['BuildingType'].str.contains('Multifamily', na=False)]
    df = df[~df['PrimaryPropertyType'].isin(exclude_types)]
    
    # Outliers SiteEUI (Logique simplifiée pour lisibilité)
    if 'SiteEUI(kBtu/sf)' in df.columns:
        df = df[(df['SiteEUI(kBtu/sf)'] > 0) & (df['SiteEUI(kBtu/sf)'] < 500)] # Seuils larges
    
    return df.copy()

# --------------------------------------------------------------------------------
# 3. PIPELINE & ENTRAÎNEMENT
# --------------------------------------------------------------------------------

def train_pipeline(df):
    
    # 1. Séparation X / y
    # Note: On garde les colonnes brutes dans X ! (YearBuilt, Latitude, etc.)
    target_col = 'SiteEnergyUse(kBtu)'
    
    # On enlève la target et les infos de fuite de données si nécessaire
    X = df.drop(columns=[target_col, 'SiteEUI(kBtu/sf)', 'SteamUse(kBtu)', 'NaturalGas(kBtu)', 'Electricity(kBtu)'], errors='ignore')
    
    # ATTENTION : Pour entraîner, on a besoin des colonnes d'énergie (Steam, Gas...) 
    # pour créer les flags (UsesSteam). Dans ton fichier d'origine, elles sont là.
    # Je les réintègre dans X pour le feature engineering, mais attention à la fuite de données directe.
    # Ici, pour créer les flags 0/1, c'est OK, mais il ne faut pas donner la consommation brute au modèle.
    cols_needed_for_flags = ['SteamUse(kBtu)', 'NaturalGas(kBtu)', 'Electricity(kBtu)']
    for c in cols_needed_for_flags:
        if c in df.columns:
            X[c] = df[c]
    
    # Log Transformation de la target (hors pipeline)
    y = np.log1p(df[target_col])
    
    # 2. Définition des colonnes
    # Ce sont les colonnes GÉNÉRÉES par FeatureEngineeringTransformer
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
    
    # Etape A : Preprocessor (agit sur les colonnes générées)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        verbose_feature_names_out=False
    )

    # Etape B : Le Grand Pipeline
    full_pipeline = Pipeline([
        ('feature_engineering', FeatureEngineeringTransformer()), # 1. On calcule les features
        ('preprocessor', preprocessor),                           # 2. On sélectionne et encode
        ('model', RandomForestRegressor(                          # 3. On prédit
            n_estimators=200, 
            max_depth=20, 
            min_samples_split=10, 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        ))
    ])

    # 5. Entraînement
    print("Entraînement du Pipeline complet...")
    # Le pipeline va appeler : transform() -> transform() -> fit()
    full_pipeline.fit(X_train, y_train)

    # 6. Eval
    print("Evaluation...")
    y_pred = full_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE (Log Scale): {rmse:.4f}")

    # 7. Sauvegarde
    joblib.dump(full_pipeline, MODEL_FILE)
    print(f"Modèle sauvegardé : {MODEL_FILE}")

if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    df = clean_data_rows(df) # On nettoie les lignes aberrantes AVANT
    train_pipeline(df)