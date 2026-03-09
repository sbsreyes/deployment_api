# API — Subscription Status Predictor

API REST para predecir si un cliente usará una suscripción,
basada en su comportamiento de compra (Shopping Behavior Dataset).

**Modelo:** Logistic Regression — Accuracy 0.84 | ROC-AUC 0.90

---

## Estructura

```
api/
│
├── model/                      ← Paquete del modelo ML
│   ├── config.yml              ← Parámetros del modelo (features, hiperparámetros)
│   ├── train_pipeline.py       ← Script de entrenamiento
│   ├── predict.py              ← Función make_prediction()
│   ├── pipeline.py             ← Pipeline sklearn
│   ├── config/core.py          ← Carga config.yml
│   ├── processing/
│   │   ├── features.py         ← Transformer personalizado (ShoppingPreprocessor)
│   │   ├── validation.py       ← Validación de inputs
│   │   └── data_manager.py     ← Carga/guarda CSVs y .pkl
│   ├── datasets/               ← CSVs de train y test
│   └── trained/                ← Pipeline entrenado (.pkl)
│
├── app/                        ← Aplicación FastAPI
│   ├── main.py                 ← Crea la app, CORS, registra rutas
│   ├── core/config.py          ← Configuración de la API (puerto, CORS)
│   ├── schemas/predict.py      ← Contratos de datos (Pydantic)
│   └── api/endpoints/
│       ├── health.py           ← GET  /api/v1/health
│       └── predict.py          ← POST /api/v1/predict
│                                  POST /api/v1/predict/single
│
├── tests/
│   └── test_api.py             ← Pruebas de integración
│
├── requirements.txt            ← Todas las dependencias
├── run.py                      ← Arranca el servidor
└── train.py                    ← Entrena y guarda el modelo
```

---

## Instalación y uso (local o EC2)

```bash
# 1. Ir a la carpeta api/
cd api/

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Entrenar el modelo (genera el .pkl)
python train.py

# 4. Arrancar la API
python run.py

# 5. Abrir documentación interactiva
# http://localhost:8001/docs
```

---

## Endpoints

| Método | Ruta                    | Descripción                        |
|--------|-------------------------|------------------------------------|
| GET    | `/api/v1/health`        | Estado de la API y versión         |
| POST   | `/api/v1/predict`       | Predicción batch (1 o N clientes)  |
| POST   | `/api/v1/predict/single`| Predicción de un solo cliente      |

---

## Ejemplo de uso

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

**Respuesta:**
```json
{
  "predictions": [
    {
      "customer_index": 0,
      "prediction": 1,
      "probability": 0.7519,
      "label": "Subscribed"
    }
  ],
  "model_version": "0.0.1",
  "total_records": 1,
  "errors": null
}
```
