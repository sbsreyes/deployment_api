"""
Validación de datos de entrada.
Usa validación manual (sin pydantic) para compatibilidad sin instalación extra.
En EC2 con requirements.txt se puede migrar a pydantic si se desea.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from model.config.core import config

# Columnas requeridas en el input (nombres originales del CSV)
REQUIRED_COLUMNS = [
    "Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases",
    "Gender", "Category", "Location", "Size", "Color", "Season",
    "Shipping Type", "Discount Applied", "Payment Method",
    "Frequency of Purchases",
]


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con nulos en columnas numéricas clave."""
    validated = input_data.copy()
    key_cols = config.model_config_.numeric_features
    present  = [c for c in key_cols if c in validated.columns]
    cols_with_na = [c for c in present if validated[c].isnull().sum() > 0]
    validated.dropna(subset=cols_with_na, inplace=True)
    return validated


def validate_inputs(
    *, input_data: pd.DataFrame
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Valida que las columnas requeridas estén presentes y sin nulos críticos.

    Returns:
        (data_válida, None) si todo está bien
        (data_vacía, mensaje_error) si hay problemas
    """
    errors = None

    # Verificar columnas mínimas presentes
    missing = [c for c in REQUIRED_COLUMNS if c not in input_data.columns]
    if missing:
        errors = f"Columnas faltantes: {missing}"
        return input_data, errors

    validated = drop_na_inputs(input_data=input_data)

    if validated.empty:
        errors = "El DataFrame quedó vacío después de eliminar nulos."

    return validated, errors

