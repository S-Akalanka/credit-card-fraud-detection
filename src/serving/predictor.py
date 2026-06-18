from pathlib import Path

import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd

from src.features.engineer import engineer
from src.utils.config import load_config
from src.utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_PATH = PROJECT_ROOT / 'logs' / 'pipelines.log'
logger = get_logger(__name__, log_file=LOG_PATH)

cfg = load_config()

class FraudPredictor:

    def __init__(self):
        self.model     = None
        self.threshold = 0.5
        self.run_name  = None

    def load(self) -> None:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name(cfg.mlflow.experiment_name)
        runs   = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.model = 'xgboost' AND tags.strategy = 'none'",
            order_by=["metrics.val_pr_auc DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError("No trained models found. Run train.py first.")

        best     = runs[0]
        run_id   = best.info.run_id
        self.run_name  = best.data.tags.get("mlflow.runName", run_id)
        self.threshold = best.data.metrics.get("best_threshold", 0.5)
        self.model     = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

        logger.info(f"Loaded: {self.run_name}  threshold={self.threshold:.3f}  "
                    f"val_pr_auc={best.data.metrics.get('val_pr_auc', 0):.4f}")

    def predict(self, features: dict, threshold=None) -> dict:

        df = pd.DataFrame([features])
        X = engineer(df)

        prob      = float(self.model.predict_proba(X)[0, 1])
        t         = threshold if threshold is not None else self.threshold
        is_fraud  = prob >= t

        return {
            "fraud_probability": round(prob, 6),
            "is_fraud"         : bool(is_fraud),
            "threshold_used"   : round(t, 4),
            "risk_level"       : "high" if prob >= 0.8 else "medium" if prob >= 0.4 else "low",
            "model"            : self.run_name,
        }
