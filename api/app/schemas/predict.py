"""
app/schemas/predict.py
======================
Define la "forma" de los datos que entran y salen de la API.

¿Por qué Pydantic aquí?
  FastAPI usa Pydantic para validar automáticamente el JSON que llega.
  Si el cliente manda un campo de más, lo ignora.
  Si falta un campo obligatorio, responde 422 con un mensaje claro.
  Si Age viene como "treinta y cinco" en vez de 35, responde 422.
  Todo esto SIN que tengas que escribir un if/else de validación.

Hay DOS partes:
  1. PredictionRequest  → lo que el cliente MANDA (entrada)
  2. PredictionResponse → lo que la API DEVUELVE (salida)
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================
# ENTRADA: Un solo registro del cliente
# ============================================================

class CustomerInput(BaseModel):
    """
    Representa UN cliente del dataset Shopping Behavior.

    Todos los campos son Optional porque:
      - La API no debe caerse si falta un campo poco importante.
      - El pipeline interno maneja los nulos con imputación.

    Los nombres usan guión_bajo para ser JSON-friendly.
    El pipeline los renombra a los nombres originales del CSV internamente.
    """

    # ── Numéricas ────────────────────────────────────────────
    Age: Optional[int] = Field(
        default=None,
        ge=18, le=100,                          # ge=greater or equal, le=less or equal
        description="Edad del cliente (18-100)",
        example=35
    )
    Purchase_Amount_USD: Optional[float] = Field(
        default=None,
        ge=0,
        description="Monto de la última compra en USD",
        example=75.0,
        alias="Purchase Amount (USD)"           # acepta también el nombre original
    )
    Review_Rating: Optional[float] = Field(
        default=None,
        ge=1.0, le=5.0,
        description="Calificación del producto (1.0 - 5.0)",
        example=4.2
    )
    Previous_Purchases: Optional[int] = Field(
        default=None,
        ge=0,
        description="Número de compras anteriores",
        example=10
    )

    # ── Categóricas ──────────────────────────────────────────
    Gender: Optional[str] = Field(
        default=None,
        description="Género del cliente",
        example="Male"
    )
    Category: Optional[str] = Field(
        default=None,
        description="Categoría del producto comprado",
        example="Clothing"
    )
    Location: Optional[str] = Field(
        default=None,
        description="Estado de EE.UU. del cliente",
        example="New York"
    )
    Size: Optional[str] = Field(
        default=None,
        description="Talla del producto (S, M, L, XL)",
        example="M"
    )
    Color: Optional[str] = Field(
        default=None,
        description="Color del producto comprado",
        example="Blue"
    )
    Season: Optional[str] = Field(
        default=None,
        description="Temporada de la compra",
        example="Summer"
    )
    Shipping_Type: Optional[str] = Field(
        default=None,
        description="Tipo de envío seleccionado",
        example="Free Shipping",
        alias="Shipping Type"
    )
    Discount_Applied: Optional[str] = Field(
        default=None,
        description="Si se aplicó descuento (Yes/No)",
        example="Yes",
        alias="Discount Applied"
    )
    Payment_Method: Optional[str] = Field(
        default=None,
        description="Método de pago utilizado",
        example="Credit Card",
        alias="Payment Method"
    )
    Frequency_of_Purchases: Optional[str] = Field(
        default=None,
        description="Frecuencia de compra del cliente",
        example="Monthly",
        alias="Frequency of Purchases"
    )

    # ── Campos extra que el modelo no usa (se ignoran en el pipeline) ────────
    # Los incluimos aquí para que la API acepte el CSV completo sin error.
    Customer_ID: Optional[int]    = Field(default=None, alias="Customer ID")
    Item_Purchased: Optional[str] = Field(default=None, alias="Item Purchased")
    Promo_Code_Used: Optional[str]= Field(default=None, alias="Promo Code Used")

    class Config:
        # Permite enviar tanto "Purchase Amount (USD)" como "Purchase_Amount_USD"
        populate_by_name = True
        # Muestra ejemplos en /docs
        schema_extra = {
            "example": {
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
                "Frequency of Purchases": "Monthly"
            }
        }


# ============================================================
# ENTRADA: Lista de registros (batch prediction)
# ============================================================

class PredictionRequest(BaseModel):
    """
    El body completo del POST /predict.

    Siempre es una lista, incluso para un solo cliente.
    Esto hace la API consistente: el cliente siempre manda una lista
    y siempre recibe una lista, sin lógica especial para N=1.
    """
    inputs: List[CustomerInput] = Field(
        description="Lista de clientes a predecir (mínimo 1)",
        min_items=1
    )

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
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
                        "Frequency of Purchases": "Monthly"
                    }
                ]
            }
        }


# ============================================================
# SALIDA: Resultado de la predicción
# ============================================================

class PredictionResult(BaseModel):
    """
    Resultado para UN cliente.

    customer_index: posición en la lista de entrada (para que el cliente
                    sepa a qué registro corresponde cada predicción).
    prediction:     0 = No suscriptor | 1 = Suscriptor
    probability:    probabilidad de ser suscriptor (0.0 → 1.0)
    label:          versión legible de prediction
    """
    customer_index: int
    prediction: int           = Field(description="0=No suscriptor, 1=Suscriptor")
    probability: float        = Field(description="Probabilidad de suscripción [0, 1]")
    label: str                = Field(description="'Subscribed' o 'Not Subscribed'")


class PredictionResponse(BaseModel):
    """
    El body completo de la respuesta del POST /predict.
    """
    predictions: List[PredictionResult]
    model_version: str        = Field(description="Versión del modelo que generó las predicciones")
    total_records: int        = Field(description="Cuántos registros fueron procesados")
    errors: Optional[str]     = Field(default=None, description="Mensaje de error si algo falló")


# ============================================================
# SALIDA: Health check
# ============================================================

class HealthResponse(BaseModel):
    """Respuesta del endpoint GET /health."""
    status: str               = Field(example="ok")
    model_version: str        = Field(example="0.0.1")
    api_version: str          = Field(example="1.0.0")
