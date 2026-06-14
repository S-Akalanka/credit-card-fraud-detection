from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.models.base import BaseModel
from src.utils.config import AppConfig
from src.utils.logger import get_logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_PATH = PROJECT_ROOT / 'logs' / 'pipelines.log'

logger = get_logger(__name__, log_file=LOG_PATH)


def run(
    cfg: AppConfig,
    model: BaseModel,
    strategy: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple:

    logger.info(f"Preprocessing - strategy: {strategy}")

    if model.name in  cfg.models.scale_features_for:
        scaler = StandardScaler()
        amount_col = -1         # Amount is always last column after dropping Time

        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()

        X_train[:, amount_col] = scaler.fit_transform(
            X_train[:, amount_col].reshape(-1, 1)
            ).ravel()
        X_val[:, amount_col] = scaler.transform(X_val[:, amount_col].reshape(-1, 1)).ravel()
        X_test[:, amount_col] = scaler.transform(X_test[:, amount_col].reshape(-1, 1)).ravel()

    class_weight_dict = None

    if strategy == "smote":
        before = y_train        # count fraud
        sm = SMOTE(
            random_state=cfg.seed,
            k_neighbors=cfg.imbalance.smote_k_neighbors,
        )
        X_train, y_train = sm.fit_resample(X_train, y_train)
        logger.info(f"SMOTE: fraud {before} -> {y_train.sum()} |"
                    f"legit: {(y_train==0).sum()}")

    elif strategy == "class_weight":
        counts = np.bincount(y_train)
        weights = len(y_train) / ( 2 * counts)
        class_weight_dict = {0: float(weights[0]), 1: float(weights[1])}
        logger.info(f"Class weights - legit: {weights[0]:.3f} | fraud: {weights[1]:.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test, class_weight_dict
