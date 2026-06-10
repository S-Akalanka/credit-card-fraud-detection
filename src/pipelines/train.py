import mlflow

from pathlib import Path

import numpy as np

from src.models.base import BaseModel
from src.pipelines import ingest, preprocess
from src.utils.config import load_config, AppConfig
from src.utils.logger import get_logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_PATH = PROJECT_ROOT / 'logs' / 'pipelines.log'

logger = get_logger(__name__, log_file= LOG_PATH)


def run_one(
    model: BaseModel,
    strategy: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    cfg: AppConfig,
) -> None:
    run_name = f"{model.name}__{strategy}"
    logger.info(f"Starting run: {run_name}")

    # Preprocess for this strategy
    (X_tr, X_v, X_te,
     y_tr, y_v, y_te,
     class_weight_dict) = preprocess.run(
        cfg, model, strategy,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"model": model.name, "strategy": strategy})

        # Fit
        model.fit(X_tr, y_tr, X_v, y_v, class_weight_dict)

        # Log hyperparams
        mlflow.log_params({
            "model": model.name,
            "strategy": strategy,
            "train_size": len(X_tr),
            **model.get_params(),
        })

        # Val metrics — used for model selection
        y_val_prob = model.predict_proba(X_v)
        y_val_pred = (y_val_prob >= 0.5).astype(int)



def run_training():

    cfg = load_config()

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    logger.info("Ingesting data...")
    X_train, X_val, X_test, y_train, y_val, y_test, _ = ingest.run(cfg)

    strategies = cfg.imbalance.strategies



if __name__ == "__main__":
    run_training()
