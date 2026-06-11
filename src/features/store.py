from pathlib import Path

import pandas as pd

from src.features.engineer import engineer
from src.utils.config import load_config
from src.utils.logger import get_logger


cfg = load_config()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT/cfg.data.raw_path
STORE_DIR = Path(PROJECT_ROOT/cfg.data.processed_path)
LOG_PATH = PROJECT_ROOT / 'logs' / 'pipelines.log'

logger = get_logger(__name__, log_file=LOG_PATH)


class FeatureStore:
    def __init__(self, store_dir: str | Path = STORE_DIR):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, version: str) -> Path:
        return self.store_dir / f"features_{version}.parquet"

    def save(self, df: pd.DataFrame, version: str = "latest") -> Path:
        path = Path(self._path(version))
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        size_mb = path.stat().st_size / 1e6
        logger.info(
            f"Feature store: saved version='{version}' | "
            f"shape={df.shape} | size={size_mb:.1f}MB | path=...{str(path)[-54:]}"
        )
        return path

    def load(self, version: str = "latest") -> pd.DataFrame:
        path = Path(self._path(version))
        if not path.exists():
            raise FileNotFoundError(
                f"Feature store version '{version}' not found at {path}. "
                f"Run: python -m src.features.store to build it."
            )
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info(f"Feature store: loaded version='{version}' | shape={df.shape}")
        return df

    def exists(self, version: str = "latest") -> bool:
        return self._path(version).exists()

    def metadata(self, version: str = "latest") -> dict:
        path = Path(self._path(version))
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        return {
            "version": version,
            "rows": len(df),
            "columns": list(df.columns),
            "size_mb": round(path.stat().st_size / 1e6, 2),
        }



def build(version: str = "latest") -> None:

    store = FeatureStore()

    if store.exists(version):
        logger.info(f"Feature store version '{version}' already exists. Skipping rebuild.")
        logger.info(f"Delete {store._path(version)} to force rebuild.")
        return

    logger.info(f"Building feature store version='{version}'...")

    df_raw = pd.read_csv(RAW_DATA_DIR)
    df_engineered = engineer(df_raw)
    store.save(df_engineered, version=version)

    meta = store.metadata(version)
    logger.info(f"Feature store ready: {meta}")


if __name__ == "__main__":
    build()
