"""
train.py  —  Entrena y guarda el modelo
=========================================
Ejecutar desde la carpeta api/:

  python train.py

Qué hace:
  1. Carga shopping_train.csv desde model/datasets/
  2. Entrena el pipeline completo (preprocesamiento + scaler + LogisticRegression)
  3. Guarda el pipeline como model/trained/modelo-subscription-output-0.0.1.pkl

Cuándo re-entrenar:
  - Cuando tengas datos nuevos (reemplaza el CSV en model/datasets/)
  - Cuando cambies hiperparámetros en model/config.yml
  - Cuando actualices la VERSION del modelo
"""

from model.train_pipeline import run_training

if __name__ == "__main__":
    run_training()
