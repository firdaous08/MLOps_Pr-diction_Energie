from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.main import app, get_db

# 1. CREATION DU MOCK 
# On crée une fausse session qui ne fait rien quand on l'appelle
mock_session = MagicMock()

# On crée une fonction qui remplace la vraie connexion BDD
def override_get_db():
    try:
        yield mock_session
    finally:
        pass

# On dit à l'API : "Quand tu as besoin de 'get_db', utilise 'override_get_db' à la place"
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# 2. TEST UNITAIRE : Vérification de la racine
def test_read_main():
    """Vérifie que l'API est bien en ligne."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API de Prédiction Énergétique en ligne !"}

# 3. TEST FONCTIONNEL : Prédiction Valide (Cas Nominal)
def test_predict_valid():
    """Vérifie qu'une prédiction fonctionne avec des données correctes."""
    # Données conformes au modèle Pydantic
    payload = {
        "BuildingType": "Supermarket/Grocery Store",
        "PrimaryPropertyType": "Supermarket/Grocery Store",
        "Neighborhood": "DOWNTOWN",
        "YearBuilt": 2010,
        "NumberofBuildings": 1.0,
        "NumberofFloors": 2.0,
        "PropertyGFATotal": 50000.0,
        "PropertyGFAParking": 0.0,
        "Latitude": 47.6,
        "Longitude": -122.3,
        "UsesSteam": 0,
        "UsesGas": 1,
        "UsesElectricity": 1,
        "IsMultiUse": 0
    }
    
    response = client.post("/predict", json=payload)
    
    # Assertions (Vérifications)
    assert response.status_code == 200
    json_data = response.json()
    
    # On vérifie que les clés importantes sont là
    assert "prediction_kbtu" in json_data
    assert "log_id" in json_data
    assert json_data["status"] == "success"
    
    # On vérifie que la prédiction est un nombre positif
    assert isinstance(json_data["prediction_kbtu"], float)
    assert json_data["prediction_kbtu"] > 0

# 4. TEST DE ROBUSTESSE : Gestion d'erreur (Cas Invalide)
def test_predict_invalid_type():
    """Vérifie que l'API rejette des données mal formées (String au lieu de Int)."""
    payload = {
        "BuildingType": "Library",
        "PrimaryPropertyType": "Library",
        "Neighborhood": "DOWNTOWN",
        "YearBuilt": "Mille neuf cent quatre vingt", # <--- ERREUR ICI (String au lieu de Int)
        "NumberofBuildings": 1,
        "NumberofFloors": 1,
        "PropertyGFATotal": 1000,
        "Latitude": 0,
        "Longitude": 0
    }
    
    response = client.post("/predict", json=payload)
    
    # On s'attend à une erreur 422 (Unprocessable Entity) générée par Pydantic
    assert response.status_code == 422