"""
app/main.py
============
Punto de entrada de la aplicación FastAPI.

Este archivo hace CUATRO cosas:

  1. Crea la instancia de FastAPI con su metadata (título, versión, docs).
  2. Configura CORS para que el dashboard (en otro puerto) pueda llamar la API.
  3. Registra todos los routers (health, predict) bajo el prefijo /api/v1.
  4. Define los eventos startup/shutdown para inicializar/liberar recursos.

¿Por qué separar main.py de los endpoints?
  Si pusiste todo en un solo archivo, cuando el proyecto crece te vuelves loco.
  La separación permite:
    - Agregar nuevos endpoints sin tocar main.py
    - Testear main.py de forma aislada
    - Múltiples personas trabajando sin conflictos de merge en git
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.api.endpoints import health, predict as predict_router

# ── Logging ──────────────────────────────────────────────────────────────────
# Configuración básica: muestra timestamp, nivel e información del módulo.
# En producción puedes enviarlo a CloudWatch, Datadog, etc.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Crear la aplicación ───────────────────────────────────────────────────────
# FastAPI genera automáticamente:
#   /docs    → Swagger UI  (interfaz interactiva para probar la API)
#   /redoc   → ReDoc       (documentación más formal)
#   /openapi.json → Especificación OpenAPI (estándar de la industria)
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
)


# ── Middleware CORS ───────────────────────────────────────────────────────────
# Sin esto, el navegador bloquea las llamadas del dashboard a la API.
# (Los navegadores tienen Same-Origin Policy por seguridad.)
#
# allow_origins:     quién puede llamar la API
# allow_methods:     qué métodos HTTP se permiten
# allow_headers:     qué headers puede mandar el cliente
# allow_credentials: si se permiten cookies (no necesario aquí)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Registrar routers ─────────────────────────────────────────────────────────
# Cada router tiene su prefijo y sus endpoints.
# El resultado final de las rutas es:
#   GET  /api/v1/health
#   POST /api/v1/predict
#   POST /api/v1/predict/single
app.include_router(
    health.router,
    prefix=settings.API_V1_STR,
)
app.include_router(
    predict_router.router,
    prefix=settings.API_V1_STR,
)


# ── Eventos de ciclo de vida ──────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """
    Se ejecuta UNA SOLA VEZ cuando el servidor arranca.

    Aquí verificamos que el modelo esté cargado correctamente.
    Si falla en startup, la API no arranca y el error queda en los logs
    (mucho mejor que un error misterioso en la primera petición).
    """
    logger.info("=" * 60)
    logger.info("Iniciando Subscription Status Predictor API")
    logger.info(f"Versión API : {settings.API_VERSION}")

    try:
        # Forzar la importación del módulo del modelo para verificar que el
        # pipeline .pkl se carga correctamente al iniciar
        from model import __version__ as model_version
        from model.predict import _subscription_pipe   # fuerza carga del pkl
        logger.info(f"Versión modelo: {model_version}")
        logger.info(f"Pipeline cargado: {type(_subscription_pipe.named_steps)}")
        logger.info("✓ Modelo listo para inferencia")
    except Exception as e:
        logger.error(f"✗ ERROR al cargar el modelo: {e}")
        raise RuntimeError(f"No se pudo cargar el modelo: {e}")

    logger.info(f"Documentación disponible en: http://0.0.0.0:{settings.PORT}/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Se ejecuta cuando el servidor se apaga."""
    logger.info("API apagándose. Recursos liberados.")


# ── Ruta raíz → redirige a /docs ──────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """
    Redirige la raíz (/) a la documentación Swagger.
    Si alguien abre http://IP:8001 en el browser, ve /docs directamente.
    """
    return RedirectResponse(url="/docs")
