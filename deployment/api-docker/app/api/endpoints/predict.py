"""
app/api/endpoints/predict.py
=============================
Endpoints de predicción.

Hay DOS endpoints en este archivo:

  POST /predict        → recibe 1 o N clientes, devuelve predicciones
  POST /predict/single → versión simplificada para llamadas desde el dashboard
                         (recibe un solo cliente directamente, sin wrapper "inputs")
Flujo interno de /predict:
  1. FastAPI valida el JSON con PredictionRequest (Pydantic)
  2. Convertimos los datos a dict (el formato que entiende make_prediction)
  3. Llamamos make_prediction() del paquete del modelo
  4. Construimos la respuesta con PredictionResponse
  5. FastAPI serializa y devuelve el JSON

Manejo de errores:
  - Si la validación falla → FastAPI responde 422 automáticamente
  - Si el modelo falla internamente → capturamos y devolvemos 500 con detalle
"""

import logging
from typing import List
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from app.schemas.predict import (
    CustomerInput,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from model.predict import make_prediction
from model import __version__ as model_version

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================
# Función auxiliar: convierte CustomerInput → dict con nombres originales
# ============================================================

def _customer_to_dict(customer: CustomerInput) -> dict:
    """
    Transforma un CustomerInput (Pydantic) en un dict con los nombres
    de columna que espera el pipeline del modelo (nombres originales del CSV).

    Ejemplo:
      CustomerInput.Purchase_Amount_USD  →  "Purchase Amount (USD)"
      CustomerInput.Shipping_Type        →  "Shipping Type"
    """
    return {
        "Age":                      customer.Age,
        "Purchase Amount (USD)":    customer.Purchase_Amount_USD,
        "Review Rating":            customer.Review_Rating,
        "Previous Purchases":       customer.Previous_Purchases,
        "Gender":                   customer.Gender,
        "Category":                 customer.Category,
        "Location":                 customer.Location,
        "Size":                     customer.Size,
        "Color":                    customer.Color,
        "Season":                   customer.Season,
        "Shipping Type":            customer.Shipping_Type,
        "Discount Applied":         customer.Discount_Applied,
        "Payment Method":           customer.Payment_Method,
        "Frequency of Purchases":   customer.Frequency_of_Purchases,
        # Campos extra que el pipeline ignora pero incluimos por si el
        # cliente manda el CSV completo
        "Customer ID":              customer.Customer_ID,
        "Item Purchased":           customer.Item_Purchased,
        "Promo Code Used":          customer.Promo_Code_Used,
    }


# ============================================================
# POST /predict  — Predicción batch (1 o N clientes)
# ============================================================

@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predecir Subscription Status (batch)",
    description=(
        "Recibe uno o más registros de clientes y devuelve para cada uno: "
        "la predicción (0/1), la probabilidad de suscripción y el label legible."
    ),
    tags=["Prediction"],
)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """
    Endpoint principal de predicción.

    - **inputs**: lista de objetos CustomerInput (mínimo 1)

    Devuelve una lista de predicciones en el mismo orden que la entrada.
    """
    logger.info(f"Recibida solicitud de predicción para {len(payload.inputs)} cliente(s)")

    # 1. Convertir lista de CustomerInput → lista de dicts
    records = [_customer_to_dict(c) for c in payload.inputs]

    # 2. Crear un DataFrame (make_prediction acepta dict o DataFrame)
    input_df = pd.DataFrame(records)

    # 3. Llamar al modelo
    try:
        result = make_prediction(input_data=input_df)
    except Exception as exc:
        logger.error(f"Error en make_prediction: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del modelo: {str(exc)}",
        )

    # 4. Verificar si el modelo reportó errores de validación
    if result.get("errors"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error de validación del modelo: {result['errors']}",
        )

    # 5. Construir la respuesta estructurada
    prediction_results: List[PredictionResult] = []
    for idx, (pred, prob) in enumerate(
        zip(result["predictions"], result["probabilities"])
    ):
        prediction_results.append(
            PredictionResult(
                customer_index=idx,
                prediction=pred,
                probability=prob,
                label="Subscribed" if pred == 1 else "Not Subscribed",
            )
        )

    logger.info(
        f"Predicciones completadas: "
        f"{sum(p.prediction for p in prediction_results)} suscriptores "
        f"de {len(prediction_results)} clientes"
    )

    return PredictionResponse(
        predictions=prediction_results,
        model_version=model_version,
        total_records=len(prediction_results),
        errors=None,
    )


# ============================================================
# POST /predict/single  — Predicción de un solo cliente
# ============================================================

@router.post(
    "/predict/single",
    response_model=PredictionResult,
    status_code=status.HTTP_200_OK,
    summary="Predecir Subscription Status (un cliente)",
    description=(
        "Recibe los datos de UN solo cliente y devuelve la predicción directamente "
        "(sin wrapper 'inputs'). Ideal para el formulario del dashboard."
    ),
    tags=["Prediction"],
)
def predict_single(customer: CustomerInput) -> PredictionResult:
    """
    Versión simplificada para predecir un único cliente.

    Manda el objeto CustomerInput directamente (sin envolver en {"inputs": [...]}).
    """
    logger.info("Recibida solicitud de predicción individual")

    input_df = pd.DataFrame([_customer_to_dict(customer)])

    try:
        result = make_prediction(input_data=input_df)
    except Exception as exc:
        logger.error(f"Error en predict_single: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del modelo: {str(exc)}",
        )

    if result.get("errors"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error de validación: {result['errors']}",
        )

    pred = result["predictions"][0]
    prob = result["probabilities"][0]

    return PredictionResult(
        customer_index=0,
        prediction=pred,
        probability=prob,
        label="Subscribed" if pred == 1 else "Not Subscribed",
    )
