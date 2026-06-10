import mlflow

from pathlib import Path

from src.pipelines import ingest
from src.utils.config import load_config
from src.utils.logger import get_logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_PATH = PROJECT_ROOT / 'logs' / 'pipelines.log'

logger = get_logger(__name__, log_file= LOG_PATH)


def run_training():

    cfg = load_config()

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    logger.info("Ingesting data...")
    ingest.run(cfg)

if __name__ == "__main__":
    run_training()
