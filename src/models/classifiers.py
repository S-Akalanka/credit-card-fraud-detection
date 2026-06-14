from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models.base import BaseModel
from src.utils.config import AppConfig
from src.utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_PATH = PROJECT_ROOT / 'logs' / 'pipelines.log'
logger = get_logger(__name__, log_file=LOG_PATH)

class BaselineModel(BaseModel):

    name = "baseline"

    def __init__(self):
        self._threshold = None
        _AMOUNT_IDX = 28

    def fit(self, X_train, y_train, X_val, y_val, class_weight_dict):
        self._thresholds = [np.percentile(X_train[y_train == 1, 13], 70),
                            np.percentile(X_train[y_train == 1, 3], 50),
                            np.percentile(X_train[y_train == 1, 12], 80),
                            np.percentile(X_train[y_train == 1, 10], 30),
                            np.percentile(X_train[y_train == 1, 9], 80)]

    def predict_proba(self, X):
        return (
                (X[:, 13] <= self._thresholds[0]) &
                (X[:, 3] >= self._thresholds[1]) &
                (X[:, 12] <= self._thresholds[2]) &
                (X[:, 10] >= self._thresholds[3]) &
                (X[:, 9] <= self._thresholds[4])
        ).astype(float)

    def get_params(self):
        return {"rule": "amount <= 5th_percentile", "threshold": self._threshold}

    @property
    def _model(self):
        raise NotImplementedError

    def log_to_mlflow(self):
        pass

class LogisticRegressionModel(BaseModel):
    name = "logistic_regression"

    def __init__(self, cfg: AppConfig):
        c = cfg.models.logistic_regression
        self._clf = LogisticRegression(
            max_iter=c.max_iter,
            solver=c.solver,
            random_state=cfg.seed,
        )

    def fit(self, X_train, y_train, X_val, y_val, class_weight_dict):
        if class_weight_dict:
            self._clf.set_params(class_weight=class_weight_dict)
        self._clf.fit(X_train, y_train)
        logger.info("Logistic regression fitted")

    def predict_proba(self, X):
        return self._clf.predict_proba(X)[:, 1]

    def get_params(self):
        return self._clf.get_params()

    @property
    def _model(self):
        return self._clf


class RandomForestModel(BaseModel):
    name = "random_forest"

    def __init__(self, cfg: AppConfig):
        c = cfg.models.random_forest
        self._clf = RandomForestClassifier(
            n_estimators=c.n_estimators,
            max_depth=c.max_depth,
            min_samples_leaf=c.min_samples_leaf,
            n_jobs=c.n_jobs,
            random_state=cfg.seed,
        )

    def fit(self, X_train, y_train, X_val, y_val, class_weight_dict):
        if class_weight_dict:
            self._clf.set_params(class_weight=class_weight_dict)
        self._clf.fit(X_train, y_train)
        logger.info(f"Random forest fitted - {self._clf.n_estimators} trees")

    def predict_proba(self, X):
        return self._clf.predict_proba(X)[:, 1]

    def get_params(self):
        return {
            "n_estimators"    : self._clf.n_estimators,
            "max_depth"       : self._clf.max_depth,
            "min_samples_leaf": self._clf.min_samples_leaf,
        }

    @property
    def _model(self):
        return self._clf


class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(self, cfg: AppConfig):
        c = cfg.models.xgboost
        self._clf = XGBClassifier(
            n_estimators=c.n_estimators,
            max_depth=c.max_depth,
            learning_rate=c.learning_rate,
            subsample=c.subsample,
            colsample_bytree=c.colsample_bytree,
            n_jobs=c.n_jobs,
            eval_metric=c.eval_metric,
            random_state=cfg.seed,
            verbosity=0,
        )

    def fit(self, X_train, y_train, X_val, y_val, class_weight_dict):
        if class_weight_dict:
            spw = class_weight_dict[1] / class_weight_dict[0]
            self._clf.set_params(scale_pos_weight=spw)
        self._clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info("XGBoost fitted")

    def predict_proba(self, X):
        return self._clf.predict_proba(X)[:, 1]

    def get_params(self):
        return {
            "n_estimators" : self._clf.n_estimators,
            "max_depth"    : self._clf.max_depth,
            "learning_rate": self._clf.learning_rate,
        }

    @property
    def _model(self):
        return self._clf


class LightGBMModel(BaseModel):
    name = "lightgbm"

    def __init__(self, cfg: AppConfig):
        c = cfg.models.lightgbm
        self._clf = LGBMClassifier(
            n_estimators=c.n_estimators,
            max_depth=c.max_depth,
            learning_rate=c.learning_rate,
            num_leaves=c.num_leaves,
            subsample=c.subsample,
            n_jobs=c.n_jobs,
            verbose=c.verbose,
            random_state=cfg.seed,
        )

    def fit(self, X_train, y_train, X_val, y_val, class_weight_dict):
        if class_weight_dict:
            self._clf.set_params(is_unbalance=True)
        self._clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
        logger.info("LightGBM fitted")

    def predict_proba(self, X):
        return self._clf.predict_proba(X)[:, 1]

    def get_params(self):
        return {
            "n_estimators" : self._clf.n_estimators,
            "max_depth"    : self._clf.max_depth,
            "learning_rate": self._clf.learning_rate,
            "num_leaves"   : self._clf.num_leaves,
        }

    @property
    def _model(self):
        return self._clf
