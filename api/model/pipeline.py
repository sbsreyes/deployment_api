"""
Pipeline completo del modelo de Subscription Status.

Arquitectura:
  ShoppingPreprocessor  →  StandardScaler  →  LogisticRegression

- ShoppingPreprocessor: todo el feature engineering (custom transformer)
- StandardScaler: necesario para que Logistic Regression converja bien
                  (fue la clave para que superara al Random Forest)
- LogisticRegression: modelo ganador (Acc 0.86, ROC-AUC 0.91)
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from model.config.core import config
from model.processing.features import ShoppingPreprocessor

subscription_pipe = Pipeline([
    (
        "preprocessor",
        ShoppingPreprocessor(freq_map=config.model_config_.freq_map),
    ),
    (
        "scaler",
        StandardScaler(),
    ),
    (
        "logistic_regression",
        LogisticRegression(
            max_iter=config.model_config_.lr_max_iter,
            C=config.model_config_.lr_c,
            random_state=config.model_config_.random_state,
            class_weight="balanced",   # maneja el desbalance de clases
        ),
    ),
])
