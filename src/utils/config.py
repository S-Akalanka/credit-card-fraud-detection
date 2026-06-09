from pathlib import Path

import yaml
from pydantic import BaseModel

class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str

class AppConfig(BaseModel):
    project_name: str
    seed: int
    mlflow: MLflowConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'config.yaml'

def load_config(path: str = CONFIG_PATH):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
