from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_path: str
    processed_path: str
    test_split: float = Field(gt=0, lt=1)
    val_split: float = Field(gt=0, lt=1)
    target_col: str
    drop_cols: List[str]

class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str

class LogisticRegressionConfig(BaseModel):
    max_iter: int  = Field(gt=0)
    solver: str


class RandomForestConfig(BaseModel):
    n_estimators: int = Field(gt=0)
    max_depth: int    = Field(gt=0)
    min_samples_leaf: int = Field(gt=0)
    n_jobs: int


class XGBoostConfig(BaseModel):
    n_estimators: int   = Field(gt=0)
    max_depth: int      = Field(gt=0)
    learning_rate: float = Field(gt=0, lt=1)
    subsample: float    = Field(gt=0, le=1)
    colsample_bytree: float = Field(gt=0, le=1)
    n_jobs: int
    eval_metric: str


class LightGBMConfig(BaseModel):
    n_estimators: int   = Field(gt=0)
    max_depth: int      = Field(gt=0)
    learning_rate: float = Field(gt=0, lt=1)
    num_leaves: int     = Field(gt=0)
    subsample: float    = Field(gt=0, le=1)
    n_jobs: int
    verbose: int


class ModelsConfig(BaseModel):
    # Models listed here receive scaled input — all others get raw features
    scale_features_for: List[str] = []
    logistic_regression: LogisticRegressionConfig
    random_forest: RandomForestConfig
    xgboost: XGBoostConfig
    lightgbm: LightGBMConfig

class ImbalanceConfig(BaseModel):
    strategies: List[str]
    smote_k_neighbors: int = Field(gt=0)

class AppConfig(BaseModel):
    project_name: str
    seed: int
    data: DataConfig
    mlflow: MLflowConfig
    models: ModelsConfig
    imbalance: ImbalanceConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'config.yaml'

def load_config(path: str = CONFIG_PATH):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
