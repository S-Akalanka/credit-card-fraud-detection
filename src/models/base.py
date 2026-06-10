from abc import ABC, abstractmethod

import mlflow
import numpy as np


class BaseModel(ABC):

    name: str

    @abstractmethod
    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            class_weight_dict: dict | None,
    ) -> None:
        """Train the model. Val set used for early stopping where supported."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud_detection probabilities (column index 1) for each sample."""
        ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @abstractmethod
    def get_params(self) -> dict:
        """Return hyperparams to log in MLflow."""
        ...

    def log_to_mlflow(self) -> None:
        """Log the fitted model artifact to the active MLflow run."""
        mlflow.sklearn.log_model(
            self._model,
            artifact_path="model",
            registered_model_name=self.name,
        )

    @property
    def _model(self):
        """Return the underlying sklearn-compatible model object."""
        raise NotImplementedError