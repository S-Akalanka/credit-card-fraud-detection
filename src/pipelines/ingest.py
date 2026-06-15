from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.store import FeatureStore
from src.utils.config import AppConfig
from src.utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_PATH = PROJECT_ROOT / 'logs' / 'pipelines.log'
logger = get_logger(__name__, log_file=LOG_PATH)

def validate(df: pd.DataFrame, cfg: AppConfig) -> None:
    required = (
        [cfg.data.target_col]
        + [f"V{i}" for i in range(1, 29)]
        + ["Hour", "Amount_log"]
    )
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in feature store: {missing}\n"
            f"Rebuild with: python -m src.features.store"
        )

    nulls = df.isnull().sum()
    if nulls.any():
        raise ValueError(f"Null values in feature store:\n{nulls[nulls > 0]}")

    class_vals = set(df[cfg.data.target_col].unique())
    if not class_vals.issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1, got: {class_vals}")

    logger.info(f"Validation passed - {len(df):,} rows, {df.shape[1]} cols")


def run(cfg: AppConfig, version: str = "latest") -> tuple:

    store = FeatureStore()

    # Auto-build store if it doesn't exist yet
    if not store.exists(version):
        logger.info(f"Feature store version '{version}' not found — building now...")
        from src.features.store import build
        build(version)

    df = store.load(version)
    validate(df, cfg)

    fraud_rate = df[cfg.data.target_col].mean() * 100
    logger.info(f"Fraud rate: {fraud_rate:.3f}%  "
                f"({df[cfg.data.target_col].sum():,} fraud / "
                f"{(df[cfg.data.target_col] == 0).sum():,} legit)")

    X = df.drop(columns=[cfg.data.target_col])
    y = df[cfg.data.target_col]

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.data.test_split,
        stratify=y,
        random_state=cfg.seed,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=cfg.data.val_split / ( 1 - cfg.data.test_split),
        stratify=y_train,
    )

    logger.info(f"Split -> train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}")

    return (
        X_train.values, X_val.values, X_test.values,
        y_train.values, y_val.values, y_test.values,
        feature_names,
    )
