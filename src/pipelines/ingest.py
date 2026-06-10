from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import AppConfig
from src.utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = get_logger(__name__, log_file=f'{PROJECT_ROOT}/logs/pipelines.log')

def validate_raw(df: pd.DataFrame, cfg: AppConfig) -> None:
    expected_cols = [ f'V{i}' for i in range(1, 29) ] + ['Time', 'Amount','Class']
    missing_cols = set(df.columns) - set(expected_cols)

    if missing_cols:
        raise ValueError(f'Missing columns in raw data: {missing_cols}')

    null_counts = df.isnull().sum()
    if null_counts.any():
        raise ValueError(f'Missing values in raw data: {null_counts[null_counts > 0]}')

    class_counts = df[cfg.data.target_col].value_counts()
    if set(class_counts.index) != {0, 1}:
        raise ValueError(f"Target column must be binary 0/1, got: {class_counts.index.tolist()}")

    logger.info(f"Validation passed - {len(df):,} rows, {df.shape[1]} cols")


def run(cfg: AppConfig) -> tuple:

    path = f'{PROJECT_ROOT}/{cfg.data.raw_path}'
    logger.info(f"Loading data from {path}")

    df = pd.read_csv(path)
    validate_raw(df, cfg)

    fraud_rate = df[cfg.data.target_col].mean() * 100
    logger.info(f"Fraud rate: {fraud_rate:.3f}%  "
                f"({df[cfg.data.target_col].sum():,} fraud / "
                f"{(df[cfg.data.target_col] == 0).sum():,} legit)")

    X = df.drop(columns=[cfg.data.target_col] + cfg.data.drop_cols)
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
