"""
run.py  —  Arranca el servidor de la API
=========================================
Ejecutar desde la carpeta api/:

  Desarrollo (recarga automática al guardar cambios):
    python run.py --reload

  Producción (sin recarga, más rápido):
    python run.py

  Puerto personalizado:
    API_PORT=9000 python run.py
"""

import sys
import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    reload = "--reload" in sys.argv

    print(f"Arrancando API en http://{settings.HOST}:{settings.PORT}")
    print(f"Documentación:  http://localhost:{settings.PORT}/docs")
    print(f"Modo reload:    {'activado' if reload else 'desactivado'}")

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=reload,
        log_level="info",
    )
