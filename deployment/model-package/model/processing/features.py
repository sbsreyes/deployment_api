"""
Transformadores personalizados sklearn para el preprocesamiento
del dataset Shopping Behavior.

"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ── Listas de categorización geográfica y de color ──────────────────────────

NORTHEAST = [
    'Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island',
    'Connecticut', 'New York', 'New Jersey', 'Pennsylvania'
]
SOUTHEAST = [
    'Delaware', 'Maryland', 'Virginia', 'West Virginia', 'North Carolina',
    'South Carolina', 'Georgia', 'Florida', 'Kentucky', 'Tennessee',
    'Alabama', 'Mississippi', 'Arkansas', 'Louisiana'
]
MIDWEST = [
    'Ohio', 'Indiana', 'Illinois', 'Michigan', 'Wisconsin', 'Minnesota',
    'Iowa', 'Missouri', 'North Dakota', 'South Dakota', 'Nebraska', 'Kansas'
]
SOUTHWEST = ['Texas', 'Oklahoma', 'New Mexico', 'Arizona']

WARM_COLORS = [
    'Red', 'Orange', 'Yellow', 'Maroon', 'Pink', 'Peach', 'Gold',
    'Magenta', 'Salmon', 'Terra cotta', 'Burgundy', 'Brown'
]
COOL_COLORS = [
    'Blue', 'Green', 'Purple', 'Teal', 'Turquoise', 'Cyan',
    'Indigo', 'Violet', 'Lavender', 'Olive'
]


def _get_region(state: str) -> str:
    if state in NORTHEAST: return 'Northeast'
    if state in SOUTHEAST: return 'Southeast'
    if state in MIDWEST:   return 'Midwest'
    if state in SOUTHWEST: return 'Southwest'
    return 'West'


def _get_color_group(color: str) -> str:
    if color in WARM_COLORS: return 'Warm'
    if color in COOL_COLORS: return 'Cool'
    return 'Neutral'


# ── Transformer principal ────────────────────────────────────────────────────

class ShoppingPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transforma el dataset crudo de Shopping Behavior en las 34 features
    usadas por el modelo de clasificación de Subscription Status.

    Pasos internos:
      1. Feature engineering (Has_Discount, Gender_Is_Male, Frequency_Numeric,
         Age_x_Freq, Total_Engagement, Region, Color Group)
      2. Drop de columnas no útiles (Customer ID, Item Purchased, etc.)
      3. One-Hot Encoding con columnas fijas (determinadas en fit)
      4. Alineación de columnas para garantizar las mismas features siempre

    Args:
        freq_map: dict de frecuencia texto → número anual
        expected_columns: lista de columnas finales (se fija en fit)
    """

    def __init__(self, freq_map: dict):
        self.freq_map = freq_map
        self.expected_columns_: List[str] = []

    # ── Internos ─────────────────────────────────────────────────────────────

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        d = X.copy()

        # Binarización
        d['Has_Discount']    = (d['Discount Applied'] == 'Yes').astype(int)
        d['Gender_Is_Male']  = (d['Gender'] == 'Male').astype(int)
        d['Frequency_Numeric'] = d['Frequency of Purchases'].map(self.freq_map).fillna(0)

        # Interacciones
        d['Age_x_Freq']       = d['Age'] * d['Frequency_Numeric']
        d['Total_Engagement'] = d['Previous Purchases'] * d['Review Rating']

        # Reducción de cardinalidad
        d['Region']       = d['Location'].map(_get_region)
        d['Color Group']  = d['Color'].map(_get_color_group)

        # Eliminar columnas originales ya transformadas o innecesarias
        drop = [
            'Customer ID', 'Item Purchased', 'Discount Applied',
            'Location', 'Color', 'Gender', 'Frequency of Purchases',
        ]
        d = d.drop(columns=[c for c in drop if c in d.columns])

        return d

    def _ohe(self, d: pd.DataFrame) -> pd.DataFrame:
        nominal_cols = [
            'Category', 'Season', 'Size', 'Shipping Type',
            'Payment Method', 'Region', 'Color Group'
        ]
        present = [c for c in nominal_cols if c in d.columns]
        d = pd.get_dummies(d, columns=present, drop_first=True, dtype=int)
        return d

    # ── API sklearn ──────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y=None):
        """Aprende las columnas que produce el OHE sobre el set de entrenamiento."""
        d = self._engineer(X)
        d = self._ohe(d)
        # Guardar columnas del train (sin el target, que ya fue extraído)
        self.expected_columns_ = [
            c for c in d.columns
            if c not in ['Subscription Status', 'Promo Code Used',
                         'Purchase Amount (USD)']
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma X y garantiza que las columnas coincidan exactamente
        con las aprendidas en fit (añade ceros para columnas faltantes,
        descarta columnas extra).
        """
        d = self._engineer(X)
        d = self._ohe(d)

        # Eliminar columnas de target/leakage si aún están presentes
        for col in ['Subscription Status', 'Promo Code Used', 'Purchase Amount (USD)']:
            if col in d.columns:
                d = d.drop(columns=[col])

        # Alinear columnas con las del entrenamiento
        d = d.reindex(columns=self.expected_columns_, fill_value=0)
        return d
