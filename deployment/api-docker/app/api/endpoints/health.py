"""
app/api/endpoints/health.py
============================
Endpoint GET /health
Responde:
  {
    "status": "ok",
    "model_version": "0.0.1",
    "api_version": "1.0.0"
  }
"""

from fastapi import APIRouter
from app.schemas.predict import HealthResponse
from app.core.config import settings

# APIRouter es como un "mini-app" de FastAPI.
# Agrupa endpoints relacionados. En main.py los incluimos todos juntos.
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Verifica que la API y el modelo estén cargados y operativos.",
    tags=["Monitoring"],
)
def health_check() -> HealthResponse:
    """
    Endpoint de verificación de estado.

    - Devuelve 200 si todo está bien.
    - Si el modelo no se pudo cargar al iniciar, este endpoint ni existe
      (la app falla antes), lo que es la señal de error más clara.
    """
    # Importamos aquí (no al top) para que el error de carga de modelo
    # aparezca con un mensaje claro si algo falla
    from model import __version__ as model_version

    return HealthResponse(
        status="ok",
        model_version=model_version,
        api_version=settings.API_VERSION,
    )
