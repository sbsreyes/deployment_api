"""
tests/test_api.py
==================
Pruebas de integración de los endpoints de la API.


Ejecutar:
  pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

# El TestClient simula un servidor FastAPI completo en memoria
client = TestClient(app)


# ── Fixture: datos de un cliente de ejemplo ───────────────────────────────────

@pytest.fixture
def single_customer():
    """Datos de un cliente válido para usar en múltiples tests."""
    return {
        "Age": 35,
        "Purchase Amount (USD)": 75.0,
        "Review Rating": 4.2,
        "Previous Purchases": 10,
        "Gender": "Male",
        "Category": "Clothing",
        "Location": "New York",
        "Size": "M",
        "Color": "Blue",
        "Season": "Summer",
        "Shipping Type": "Free Shipping",
        "Discount Applied": "Yes",
        "Payment Method": "Credit Card",
        "Frequency of Purchases": "Monthly",
    }


# ── Tests del endpoint /health ────────────────────────────────────────────────

def test_health_returns_200():
    """El health check debe devolver 200 OK."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_response_structure():
    """La respuesta del health check debe tener los campos esperados."""
    response = client.get("/api/v1/health")
    data = response.json()
    assert "status" in data
    assert "model_version" in data
    assert "api_version" in data
    assert data["status"] == "ok"


# ── Tests del endpoint /predict ───────────────────────────────────────────────

def test_predict_returns_200(single_customer):
    """Una predicción válida debe devolver 200."""
    payload = {"inputs": [single_customer]}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200


def test_predict_response_structure(single_customer):
    """La respuesta debe tener la estructura correcta."""
    payload = {"inputs": [single_customer]}
    response = client.post("/api/v1/predict", json=payload)
    data = response.json()

    assert "predictions" in data
    assert "model_version" in data
    assert "total_records" in data
    assert data["total_records"] == 1


def test_predict_prediction_is_binary(single_customer):
    """La predicción debe ser 0 o 1."""
    payload = {"inputs": [single_customer]}
    response = client.post("/api/v1/predict", json=payload)
    data = response.json()

    pred = data["predictions"][0]["prediction"]
    assert pred in [0, 1]


def test_predict_probability_range(single_customer):
    """La probabilidad debe estar entre 0 y 1."""
    payload = {"inputs": [single_customer]}
    response = client.post("/api/v1/predict", json=payload)
    data = response.json()

    prob = data["predictions"][0]["probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_label_matches_prediction(single_customer):
    """El label debe coincidir con la predicción numérica."""
    payload = {"inputs": [single_customer]}
    response = client.post("/api/v1/predict", json=payload)
    data = response.json()

    result = data["predictions"][0]
    if result["prediction"] == 1:
        assert result["label"] == "Subscribed"
    else:
        assert result["label"] == "Not Subscribed"


def test_predict_batch_multiple_customers(single_customer):
    """El endpoint debe procesar múltiples clientes en una sola llamada."""
    customer_2 = {**single_customer, "Age": 22, "Previous Purchases": 1}
    payload = {"inputs": [single_customer, customer_2]}

    response = client.post("/api/v1/predict", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2


def test_predict_invalid_payload_returns_422():
    """Un payload vacío debe devolver 422 Unprocessable Entity."""
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 422


def test_predict_age_out_of_range():
    """Una edad fuera del rango válido (ge=18, le=100) debe devolver 422."""
    bad_customer = {
        "Age": 5,   # menor a 18, viola ge=18
        "Gender": "Male",
    }
    response = client.post("/api/v1/predict", json={"inputs": [bad_customer]})
    # FastAPI valida con Pydantic y devuelve 422 automáticamente
    assert response.status_code == 422


# ── Tests del endpoint /predict/single ───────────────────────────────────────

def test_predict_single_returns_200(single_customer):
    """El endpoint single debe devolver 200."""
    response = client.post("/api/v1/predict/single", json=single_customer)
    assert response.status_code == 200


def test_predict_single_response_is_one_result(single_customer):
    """El endpoint single devuelve un objeto (no una lista)."""
    response = client.post("/api/v1/predict/single", json=single_customer)
    data = response.json()

    # Es un objeto directo, no una lista con 'predictions'
    assert "prediction" in data
    assert "probability" in data
    assert "label" in data
    assert data["customer_index"] == 0
