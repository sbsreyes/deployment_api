# deployment/

Contiene los dos pasos del despliegue del modelo de predicción
de **Subscription Status** en AWS EC2 con Docker.

---

## Estructura

```
deployment/
│
├── model-package/                  ← PASO 1: Empaquetar el modelo
│   ├── tox.ini                     ← automatización (train + test)
│   ├── setup.py                    ← metadatos del paquete
│   ├── pyproject.toml
│   ├── MANIFEST.in
│   ├── model/                      ← código del modelo ML
│   │   ├── config.yml              ← features, hiperparámetros
│   │   ├── pipeline.py             ← sklearn Pipeline
│   │   ├── train_pipeline.py       ← script de entrenamiento
│   │   ├── predict.py              ← función make_prediction()
│   │   ├── config/core.py          ← carga config.yml
│   │   ├── processing/
│   │   │   ├── features.py         ← ShoppingPreprocessor (transformer)
│   │   │   ├── validation.py       ← validación de inputs
│   │   │   └── data_manager.py     ← carga CSVs y .pkl
│   │   ├── datasets/               ← train y test CSV
│   │   └── trained/                ← .pkl generado por tox
│   ├── requirements/
│   │   ├── requirements.txt
│   │   └── test_requirements.txt
│   └── tests/
│       ├── conftest.py
│       └── test_prediction.py
│
└── api-docker/                     ← PASO 2: Desplegar como API con Docker
    ├── Dockerfile                  ← imagen de la API
    ├── docker-compose.yml          ← orquestación del contenedor
    ├── .dockerignore
    ├── requirements.txt            ← dependencias FastAPI (sin el modelo)
    ├── packages/                   ← ← ← AQUÍ VA EL .whl del Paso 1
    │   └── model_subscription-0.0.1-py3-none-any.whl
    ├── app/
    │   ├── main.py                 ← crea la app FastAPI
    │   ├── core/config.py          ← configuración (puerto, CORS)
    │   ├── schemas/predict.py      ← contratos de datos (Pydantic)
    │   └── api/endpoints/
    │       ├── health.py           ← GET  /api/v1/health
    │       └── predict.py          ← POST /api/v1/predict
    └── tests/
        └── test_api.py             ← pruebas de integración
```

---

## Flujo completo en EC2

```
┌─────────────────────────────────────────────────────────────┐
│  PASO 1 — model-package/                                    │
│                                                             │
│  tox run -e train          → entrena el modelo              │
│  tox run -e test_package   → pruebas unitarias              │
│  python3 -m build          → genera dist/*.whl              │
│                                                             │
│  Artefacto: model_subscription-0.0.1-py3-none-any.whl      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                  cp dist/*.whl
                  ../api-docker/packages/
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  PASO 2 — api-docker/                                       │
│                                                             │
│  docker-compose up --build                                  │
│    └── pip install requirements.txt                         │
│    └── pip install packages/*.whl   ← instala el modelo     │
│    └── uvicorn app.main:app :8001                           │
│                                                             │
│  API disponible: http://IP_EC2:8001/docs                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Comandos en EC2

### Preparación (una sola vez)

```bash
# Actualizar e instalar herramientas
sudo apt update
sudo apt install -y python3-pip python3.12-venv zip unzip git
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo usermod -aG docker ubuntu && newgrp docker

# Instalar tox
pip install tox
PATH=$PATH:/home/ubuntu/.local/bin
```

### PASO 1 — Empaquetar el modelo

```bash
cd deployment/model-package/

# Ambiente virtual para tox
python3 -m venv env-tox
source env-tox/bin/activate
pip install tox

# Entrenar  ← screenshot requerido
tox run -e train

# Pruebas   ← screenshot requerido
tox run -e test_package

# Construir el paquete  ← screenshot de dist/ requerido
python3 -m pip install --upgrade build
python3 -m build

# Copiar el .whl a la API
cp dist/model_subscription-0.0.1-py3-none-any.whl \
   ../api-docker/packages/
```

### PASO 2 — Desplegar con Docker

```bash
cd ../api-docker/

# Construir imagen y arrancar  ← screenshot requerido
docker-compose up --build

# En segundo plano:
docker-compose up -d --build

# Ver logs:
docker-compose logs -f
```

### Verificar que funciona

```bash
# Health check
curl http://localhost:8001/api/v1/health

# Predicción
curl -X POST http://localhost:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
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
    }]
  }'
```

**Respuesta esperada:**
```json
{
  "predictions": [{
    "customer_index": 0,
    "prediction": 1,
    "probability": 0.7519,
    "label": "Subscribed"
  }],
  "model_version": "0.0.1",
  "total_records": 1,
  "errors": null
}
```

### Comandos útiles de Docker

```bash
docker ps                    # ver contenedores corriendo
docker-compose down          # detener
docker-compose logs -f       # ver logs en tiempo real
docker images                # ver imágenes construidas
```

### Seguridad en EC2

Abrir el puerto **8001** en el Security Group:
- Tipo: Custom TCP
- Puerto: 8001
- Source: 0.0.0.0/0
