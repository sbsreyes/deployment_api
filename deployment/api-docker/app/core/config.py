"""
app/core/config.py
==================
Centraliza TODA la configuración de la API.

"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    # ── Identidad de la API ──────────────────────────────────────────────────
    API_TITLE: str = "Subscription Status Predictor API"
    API_DESCRIPTION: str = (
        "API para predecir si un cliente de retail usará una suscripción, "
        "basado en su comportamiento de compra. "
        "Modelo: Logistic Regression — Accuracy 0.84 | ROC-AUC 0.90"
    )
    API_VERSION: str = "1.0.0"

    # ── Prefijo de rutas ─────────────────────────────────────────────────────
    # Todas las rutas quedan bajo /api/v1/...
    # Así si mañana cambias el modelo, creas /api/v2/ sin romper la v1.
    API_V1_STR: str = "/api/v1"

    # ── Servidor ─────────────────────────────────────────────────────────────
    # Lee del entorno, si no hay variable usa el default.
    HOST: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    PORT: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8001")))

    # ── CORS (Cross-Origin Resource Sharing) ─────────────────────────────────
    # CORS controla qué dominios pueden llamar la API desde un navegador.
    # Sin esto, el dashboard en el browser sería bloqueado por el browser.
    #
    # En desarrollo: "*" permite cualquier origen (cómodo para probar).
    # En producción: pon solo tu dominio real, ej: ["https://mi-dashboard.com"]
    ALLOWED_ORIGINS: List[str] = field(default_factory=lambda: [
        "*",                         # permite cualquier dominio (dev/demo)
        "http://localhost:3000",     # React dev server
        "http://localhost:8501",     # Streamlit dashboard
        "http://localhost:8050",     # Dash dashboard
    ])

    # ── Documentación automática ─────────────────────────────────────────────
    # FastAPI genera /docs (Swagger UI) y /redoc automáticamente.
    # En producción podrías desactivarlos por seguridad.
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"


# Instancia global — se importa en toda la app
settings = Settings()
