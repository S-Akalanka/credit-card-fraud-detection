from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_path: str
    test_split: float = Field(gt=0, lt=1)
    val_split: float = Field(gt=0, lt=1)
    target_col: str
    drop_cols: List[str]

class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str

class ImbalanceConfig(BaseModel):
    strategies: List[str]
    smote_k_neighbors: int = Field(gt=0)

class AppConfig(BaseModel):
    project_name: str
    seed: int
    data: DataConfig
    mlflow: MLflowConfig
    imbalance: ImbalanceConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'config.yaml'

def load_config(path: str = CONFIG_PATH):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
