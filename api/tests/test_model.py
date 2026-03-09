"""
tests/test_model.py
====================
Pruebas unitarias del modelo — equivalente al test_prediction.py del taller.

Prueban make_prediction() directamente, sin HTTP ni servidor.
Se ejecutan con:
    tox run -e test_model
"""

import pytest
from model.config.core import config
from model.processing.data_manager import load_dataset
from model.predict import make_prediction


@pytest.fixture()
def sample_input_data():
    """Carga el CSV de test para las pruebas."""
    return load_dataset(file_name=config.app_config.test_data_file)


def test_prediction_count(sample_input_data):
    """El número de predicciones debe igualar el número de filas de entrada."""
    result = make_prediction(input_data=sample_input_data)

    assert result.get("predictions") is not None
    assert result.get("errors") is None
    assert len(result["predictions"]) == len(sample_input_data)


def test_predictions_are_binary(sample_input_data):
    """Todas las predicciones deben ser 0 o 1."""
    result = make_prediction(input_data=sample_input_data)

    assert all(p in [0, 1] for p in result["predictions"])


def test_probabilities_in_range(sample_input_data):
    """Las probabilidades deben estar entre 0 y 1."""
    result = make_prediction(input_data=sample_input_data)

    assert all(0.0 <= p <= 1.0 for p in result["probabilities"])


def test_model_version_present(sample_input_data):
    """El resultado debe incluir la versión del modelo."""
    result = make_prediction(input_data=sample_input_data)

    assert result.get("version") is not None
    assert result["version"] == "0.0.1"


def test_single_record_prediction():
    """Verifica que la predicción funciona con un único registro."""
    import pandas as pd

    single = pd.DataFrame([{
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
        "Promo Code Used": "Yes",
        "Payment Method": "Credit Card",
        "Frequency of Purchases": "Monthly",
        "Item Purchased": "T-shirt",
        "Customer ID": 999,
    }])

    result = make_prediction(input_data=single)

    assert result.get("errors") is None
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] in [0, 1]
    assert 0.0 <= result["probabilities"][0] <= 1.0

    print(f"\n  Predicción: {result['predictions'][0]}")
    print(f"  Probabilidad: {result['probabilities'][0]}")
    print(f"  Versión: {result['version']}")
