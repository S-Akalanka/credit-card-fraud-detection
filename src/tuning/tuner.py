from pathlib import Path

import mlflow
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.evaluation.metrics import compute_metrics
from src.models.classifiers import XGBoostModel
from src.pipelines import ingest, preprocess
from src.utils.config import load_config
from src.utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_PATH = PROJECT_ROOT / 'logs' / 'pipelines.log'
logger = get_logger(__name__, log_file=LOG_PATH)

def sample_params(trial: optuna.Trial) -> dict:
    """
    learning_rate: log scale - 0.01 vs 0.05 matters more than 0.25 vs 0.29
    reg_alpha/lambda: log scale - regularization effect is exponential
    scale_pos_weight: tuned around n_legit/n_fraud ~ 577, +-buffer for Optuna to explore
    gamma: minimum loss to make a split - higher = more conservative tree
    """
    return {
        "n_estimators"     : trial.suggest_int("n_estimators", 200, 1000, step=50),
        "max_depth"        : trial.suggest_int("max_depth", 3, 8),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
        "gamma"            : trial.suggest_float("gamma", 0, 5),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "scale_pos_weight" : trial.suggest_float("scale_pos_weight", 400, 700),
    }

def make_objective(X_train, y_train, X_val, y_val, cfg):

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)

        model = XGBClassifier(
            **params,
            eval_metric="aucpr",
            random_state=cfg.seed,
            verbosity=0,
            n_jobs=-1,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_prob = model.predict_proba(X_val)[:, 1]
        val_pr_auc = average_precision_score(y_val, y_prob)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metric("val_pr_auc", val_pr_auc)

        return val_pr_auc

    return objective

def run_tuning(n_trials: int = 75) -> None:

    cfg = load_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, X_val, X_test, y_train, y_val, y_test, _ = ingest.run(cfg)

    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess.run(
        cfg, XGBoostModel(cfg), "none",
        X_train, X_val, X_test,
        y_train, y_val, y_test,
    )

    logger.info(f"Starting Optuna - {n_trials} trials")

    with mlflow.start_run(run_name="xgboost__optuna_tuning"):
        
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=cfg.seed),
            study_name="xgboost_fraud",
        )
        study.optimize(
            make_objective(X_train, y_train, X_val, y_val, cfg),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_val_pr_auc = study.best_value
        logger.info(f"Best val_pr_auc : {best_val_pr_auc:.4f}")
        logger.info(f"Best params     : {best_params}")

        # Final model on train+val combined
        X_trainval = np.concatenate([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])

        final_model = XGBClassifier(
            **best_params,
            eval_metric="aucpr",
            random_state=cfg.seed,
            verbosity=0,
            n_jobs=-1,
        )
        final_model.fit(X_trainval, y_trainval)

        # Test evaluation
        y_prob = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        test_metrics = compute_metrics(y_test, y_pred, y_prob)

        logger.info(f"Test PR-AUC : {test_metrics['pr_auc']:.4f}")
        logger.info(f"Test F1     : {test_metrics['f1']:.4f}")

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("val_pr_auc", best_val_pr_auc)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("n_trials", n_trials)
        mlflow.set_tags({"model": "xgboost", "strategy": "optuna_tuned"})

        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=f"{cfg.project_name}__xgboost__tuned",
        )

        logger.info("Done. Run: mlflow ui --port 5000 to compare tuned vs untuned")


if __name__ == "__main__":
    run_tuning()
