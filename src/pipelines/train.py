import mlflow
from src.utils.config import load_config

def run_training():

    cfg = load_config()

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)


if __name__ == "__main__":
    run_training()
