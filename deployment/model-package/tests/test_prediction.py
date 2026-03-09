"""
tests/test_prediction.py
=========================
Pruebas unitarias de make_prediction().
Se ejecutan con:  tox run -e test_package
"""

import pandas as pd
from model.predict import make_prediction


def test_prediction_count(sample_input_data):
    result = make_prediction(input_data=sample_input_data)
    assert result["errors"] is None
    assert len(result["predictions"]) == len(sample_input_data)


def test_predictions_are_binary(sample_input_data):
    result = make_prediction(input_data=sample_input_data)
    assert all(p in [0, 1] for p in result["predictions"])


def test_probabilities_in_range(sample_input_data):
    result = make_prediction(input_data=sample_input_data)
    assert all(0.0 <= p <= 1.0 for p in result["probabilities"])


def test_model_version(sample_input_data):
    result = make_prediction(input_data=sample_input_data)
    assert result["version"] == "0.0.1"


def test_single_record():
    single = pd.DataFrame([{
        "Age": 35, "Purchase Amount (USD)": 75.0, "Review Rating": 4.2,
        "Previous Purchases": 10, "Gender": "Male", "Category": "Clothing",
        "Location": "New York", "Size": "M", "Color": "Blue", "Season": "Summer",
        "Shipping Type": "Free Shipping", "Discount Applied": "Yes",
        "Payment Method": "Credit Card", "Frequency of Purchases": "Monthly",
        "Item Purchased": "T-shirt", "Customer ID": 999, "Promo Code Used": "Yes",
    }])
    result = make_prediction(input_data=single)

    assert result["errors"] is None
    assert result["predictions"][0] in [0, 1]
    assert 0.0 <= result["probabilities"][0] <= 1.0

    print(f"\n  Predicción:   {result['predictions'][0]}")
    print(f"  Probabilidad: {result['probabilities'][0]}")
