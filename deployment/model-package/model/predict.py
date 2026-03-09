"""
Función de inferencia del modelo de Subscription Status.

Esta función es la que consume la API:
  input: dict o DataFrame con las columnas del CSV original
  output: dict con predictions (0/1), probabilities y versión
"""

import typing as t
import pandas as pd

from model import __version__ as _version
from model.config.core import config
from model.processing.data_manager import load_pipeline
from model.processing.validation import validate_inputs

# ── Cargar el pipeline al importar el módulo ────────────────────────────────
_pipeline_file = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_subscription_pipe = load_pipeline(file_name=_pipeline_file)


# ── Mapa de nombres: JSON de la API → columnas del CSV ──────────────────────
_COLUMN_RENAME = {
    "Review_Rating":          "Review Rating",
    "Previous_Purchases":     "Previous Purchases",
    "Purchase_Amount_USD":    "Purchase Amount (USD)",
    "Shipping_Type":          "Shipping Type",
    "Discount_Applied":       "Discount Applied",
    "Payment_Method":         "Payment Method",
    "Frequency_of_Purchases": "Frequency of Purchases",
}


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """
    Realiza predicciones de Subscription Status.

    Args:
        input_data: dict o DataFrame con los campos crudos del cliente
                    (mismos nombres que el CSV original o con guiones bajos)

    Returns:
        dict con:
          - predictions: lista de 0 (No) o 1 (Yes)
          - probabilities: probabilidad de ser suscriptor [0.0, 1.0]
          - version: versión del modelo
          - errors: None o mensaje de error de validación
    """
    data = pd.DataFrame(input_data)

    # Renombrar columnas snake_case → nombres originales del CSV
    data = data.rename(columns=_COLUMN_RENAME)

    # Validar inputs
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "probabilities": None,
               "version": _version, "errors": errors}

    if not errors:
        predictions  = _subscription_pipe.predict(X=validated_data)
        probabilities = _subscription_pipe.predict_proba(X=validated_data)[:, 1]

        results = {
            "predictions":   [int(p) for p in predictions],
            "probabilities": [round(float(p), 4) for p in probabilities],
            "version":       _version,
            "errors":        None,
        }

    return results
