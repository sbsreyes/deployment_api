from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# ── Rutas del paquete ────────────────────────────────────────────────────────
PACKAGE_ROOT      = Path(__file__).resolve().parent.parent
ROOT              = PACKAGE_ROOT.parent
CONFIG_FILE_PATH  = PACKAGE_ROOT / "config.yml"
DATASET_DIR       = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained"


# ── Esquemas de configuración (dataclasses, sin dependencias externas) ───────

@dataclass
class AppConfig:
    package_name: str
    train_data_file: str
    test_data_file: str
    pipeline_save_file: str


@dataclass
class ModelConfig:
    target: str
    drop_cols: List[str]
    leakage_cols: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    model_features: List[str]
    freq_map: Dict[str, int]
    qual_mappings: Dict[str, int]
    pipeline_name: str
    test_size: float
    random_state: int
    lr_max_iter: int
    lr_c: float


@dataclass
class Config:
    app_config: AppConfig
    model_config_: ModelConfig


# ── Funciones de carga ───────────────────────────────────────────────────────

def find_config_file() -> Path:
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise FileNotFoundError(f"Config no encontrado en: {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> dict:
    if not cfg_path:
        cfg_path = find_config_file()
    with open(cfg_path, "r") as conf_file:
        return yaml.safe_load(conf_file)


def create_and_validate_config(raw: Optional[dict] = None) -> Config:
    if raw is None:
        raw = fetch_config_from_yaml()
    return Config(
        app_config=AppConfig(
            package_name=raw["package_name"],
            train_data_file=raw["train_data_file"],
            test_data_file=raw["test_data_file"],
            pipeline_save_file=raw["pipeline_save_file"],
        ),
        model_config_=ModelConfig(
            target=raw["target"],
            drop_cols=raw["drop_cols"],
            leakage_cols=raw["leakage_cols"],
            numeric_features=raw["numeric_features"],
            categorical_features=raw["categorical_features"],
            model_features=raw["model_features"],
            freq_map=raw["freq_map"],
            qual_mappings=raw["qual_mappings"],
            pipeline_name=raw["pipeline_name"],
            test_size=float(raw["test_size"]),
            random_state=int(raw["random_state"]),
            lr_max_iter=int(raw["lr_max_iter"]),
            lr_c=float(raw["lr_c"]),
        ),
    )


config = create_and_validate_config()
