"""
Script de entrenamiento del modelo de Subscription Status.

Ejecutar desde la raíz del paquete:
    python model/train_pipeline.py

O mediante tox:
    tox run -e train
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from model.config.core import config
from model.pipeline import subscription_pipe
from model.processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    """Entrena el pipeline completo y lo persiste como .pkl."""

    print("=" * 60)
    print("ENTRENAMIENTO — Subscription Status Classifier")
    print("=" * 60)

    # 1. Cargar datos
    data = load_dataset(file_name=config.app_config.train_data_file)
    print(f"  Datos cargados: {data.shape}")

    # 2. Mapear target a 0/1
    data[config.model_config_.target] = (
        data[config.model_config_.target] == "Yes"
    ).astype(int)

    # 3. Separar X e y  (el preprocesamiento ocurre DENTRO del pipeline)
    X = data.drop(columns=[config.model_config_.target])
    y = data[config.model_config_.target]

    print(f"  Balance: {y.mean():.2%} suscriptores")

    # 4. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.model_config_.test_size,
        random_state=config.model_config_.random_state,
        stratify=y,
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 5. Entrenar
    subscription_pipe.fit(X_train, y_train)

    # 6. Evaluar
    y_pred = subscription_pipe.predict(X_test)
    y_prob = subscription_pipe.predict_proba(X_test)[:, 1]

    print("\n  Métricas:")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1-Score : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("\n" + classification_report(
        y_test, y_pred, target_names=["No Subscribed", "Subscribed"]
    ))

    # 7. Guardar
    save_pipeline(pipeline_to_persist=subscription_pipe)
    print("✓ Pipeline guardado correctamente.")


if __name__ == "__main__":
    run_training()
