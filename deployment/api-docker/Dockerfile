# ============================================================
# Dockerfile — Subscription Status Predictor API
# ============================================================
#
# FLUJO DE BUILD:
#
#   1. Parte de una imagen base de Python
#   2. Instala las dependencias de la API (requirements.txt)
#   3. Instala el modelo desde el .whl (packages/)
#   4. Copia el código de la app
#   5. Expone el puerto y arranca uvicorn
#
# CÓMO CONSTRUIR Y CORRER:
#
#   # Opción A — Docker solo:
#   docker build -t subscription-api .
#   docker run -p 8001:8001 subscription-api
#
#   # Opción B — Docker Compose (recomendado):
#   docker-compose up --build
#
# ============================================================

# ── Imagen base ───────────────────────────────────────────────────────────────
# python:3.11-slim es una imagen minimalista de Python 3.11.
# "slim" = sin compiladores ni herramientas de desarrollo → imagen más pequeña.
FROM python:3.11-slim

# ── Variables de entorno ──────────────────────────────────────────────────────
# PYTHONDONTWRITEBYTECODE=1  → no genera archivos .pyc (innecesarios en Docker)
# PYTHONUNBUFFERED=1         → los logs aparecen en tiempo real (sin buffer)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8001

# ── Directorio de trabajo dentro del contenedor ───────────────────────────────
# Todo el código de la app vive en /app dentro del contenedor.
WORKDIR /app

# ── Instalar dependencias del sistema ────────────────────────────────────────
# Actualizamos pip primero para evitar warnings de versión desactualizada.
RUN pip install --upgrade pip

# ── Instalar dependencias de la API ──────────────────────────────────────────
# Copiamos SOLO requirements.txt primero (antes del código).
# Esto aprovecha el cache de Docker: si el código cambia pero las dependencias
# no, Docker no reinstala todo — solo reconstruye desde COPY . .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Instalar el paquete del modelo desde el .whl ──────────────────────────────
# El .whl se copia desde la carpeta packages/ local.
# Este archivo fue generado en el Paso 1 con: python3 -m build
COPY packages/ packages/
RUN pip install --no-cache-dir packages/model_subscription-0.0.1-py3-none-any.whl

# ── Copiar el código de la aplicación ────────────────────────────────────────
# Lo copiamos AL FINAL para maximizar el uso del cache de Docker.
# Si solo cambias el código Python (no requirements ni el .whl),
# Docker reutiliza las capas anteriores y el rebuild es instantáneo.
COPY app/ app/

# ── Exponer el puerto ─────────────────────────────────────────────────────────
# EXPOSE es documentación — indica qué puerto usa el contenedor.
# El mapeo real se hace en docker run -p o en docker-compose.yml.
EXPOSE 8001

# ── Comando de arranque ───────────────────────────────────────────────────────
# uvicorn arranca la app FastAPI.
# --host 0.0.0.0 → escucha en todas las interfaces (necesario en Docker)
# --port 8001    → puerto dentro del contenedor
# --workers 1    → un worker (suficiente para el prototipo)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
