import pandas as pd
from sqlalchemy import Column, Integer, Float, String
from app.database import Base, engine, PredictionLog
import os

# 1. Définition de la table pour le Dataset
class BuildingData(Base):
    __tablename__ = "building_data"
    id = Column(Integer, primary_key=True, index=True)
    OSEBuildingID = Column(Integer)
    DataYear = Column(Integer)
    BuildingType = Column(String)
    PrimaryPropertyType = Column(String)
    PropertyName = Column(String)
    Address = Column(String)
    ZipCode = Column(Float)
    Neighborhood = Column(String)
    YearBuilt = Column(Integer)
    NumberofBuildings = Column(Float)
    NumberofFloors = Column(Float)
    PropertyGFATotal = Column(Float)
    SiteEnergyUse_kBtu = Column(Float)

# 2. Création des tables dans PostgreSQL
print("Création des tables (building_data + prediction_logs)...")
Base.metadata.create_all(bind=engine)
print(" Tables créées !")

# 3. Insertion du CSV
csv_path = "data/2016_Building_Energy_Benchmarking.csv" # Vérifie que le CSV est bien là !

if os.path.exists(csv_path):
    print(" Chargement du CSV...")
    df = pd.read_csv(csv_path)
    
    # Nettoyage des noms de colonnes pour SQL
    df = df.rename(columns={'SiteEnergyUse(kBtu)': 'SiteEnergyUse_kBtu'})
    
    # On garde uniquement les colonnes utiles
    cols_to_keep = [
        'OSEBuildingID', 'DataYear', 'BuildingType', 'PrimaryPropertyType', 
        'PropertyName', 'Address', 'ZipCode', 'Neighborhood', 
        'YearBuilt', 'NumberofBuildings', 'NumberofFloors', 
        'PropertyGFATotal', 'SiteEnergyUse_kBtu'
    ]
    # On filtre (si des colonnes manquent, on ignore pour l'exercice)
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df_filtered = df[available_cols].copy()
    
    print(f"Insertion de {len(df_filtered)} lignes dans la base...")
    df_filtered.to_sql('building_data', con=engine, if_exists='replace', index=False)
    print(" Dataset inséré avec succès !")
else:
    print(f" Attention : Je ne trouve pas le fichier {csv_path}. Seules les tables vides ont été créées.")