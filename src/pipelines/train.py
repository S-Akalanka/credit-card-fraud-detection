import mlflow

from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.models.base import BaseModel
from src.models.classifiers import BaselineModel, LogisticRegressionModel, RandomForestModel, XGBoostModel, \
    LightGBMModel
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
        val_metrics = compute_metrics(y_v, y_val_pred, y_val_prob)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # Test metrics — honest final performance, not used for selection
        y_test_prob = model.predict_proba(X_te)
        y_test_pred = (y_test_prob >= 0.5).astype(int)
        test_metrics = compute_metrics(y_te, y_test_pred, y_test_prob)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Log model artifact
        model.log_to_mlflow()

        logger.info(
            f"  val  PR-AUC={val_metrics['pr_auc']:.4f} "
            f"F1={val_metrics['f1']:.4f} "
            f"recall={val_metrics['recall']:.4f}"
        )
        logger.info(f"  test PR-AUC={test_metrics['pr_auc']:.4f}")


def run_training():

    cfg = load_config()

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    logger.info("Ingesting data...")
    X_train, X_val, X_test, y_train, y_val, y_test, _ = ingest.run(cfg)

    strategies = cfg.imbalance.strategies

    # Baseline — strategy doesn't apply
    run_one(
        BaselineModel(), "none",
        X_train, X_val, X_test,
        y_train, y_val, y_test, cfg,
    )

    # All trained models × all strategies
    model_classes = [
        LogisticRegressionModel,
        RandomForestModel,
        XGBoostModel,
        LightGBMModel,
    ]

    for ModelClass in tqdm(model_classes, desc="Models"):
        for strategy in tqdm(strategies, desc="Strategies", leave=False):
            run_one(
                ModelClass(cfg),
                strategy,
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                cfg,
            )

    logger.info("All experiments complete. View: mlflow ui --port 5000")


if __name__ == "__main__":
    run_training()
