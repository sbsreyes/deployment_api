import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Carga un CSV desde la carpeta datasets del paquete."""
    return pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Guarda el pipeline entrenado con nombre versionado."""
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    print(f"  → Guardado: {save_path}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Carga el pipeline guardado."""
    return joblib.load(filename=TRAINED_MODEL_DIR / file_name)


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """Elimina versiones anteriores del pipeline (.pkl)."""
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
