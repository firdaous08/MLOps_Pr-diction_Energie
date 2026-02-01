from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    """Vérifie que la route '/' répond bien"""
    response = client.get("/")
    assert response.status_code == 200
    # On vérifie juste que le message est le bon
    assert response.json() == {"message": "API de Prédiction Énergétique en ligne !"}
    
def test_predict_endpoint():
    """Vérifie qu'une prédiction fonctionne avec des données bidons"""
    data = {
        "BuildingType": "Office",
        "PrimaryPropertyType": "Small Office",
        "Neighborhood": "DOWNTOWN",
        "YearBuilt": 1990,
        "NumberofBuildings": 1,
        "NumberofFloors": 3,
        "PropertyGFATotal": 25000,
        "PropertyGFAParking": 0,
        "Latitude": 47.61,
        "Longitude": -122.33,
        "UsesSteam": 0,
        "UsesGas": 0,
        "UsesElectricity": 1,
        "IsMultiUse": 0
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    # On vérifie qu'on reçoit bien une prédiction
    assert "prediction_kbtu" in response.json()