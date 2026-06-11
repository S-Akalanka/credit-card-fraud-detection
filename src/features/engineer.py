from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_PATH = PROJECT_ROOT / 'logs' / 'pipelines.log'
logger = get_logger(__name__, log_file=LOG_PATH)

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time-based features
    df["Hour"]       = (df["Time"] / 3600) % 24

    # Amount transform
    # WHY +1: log(0) is undefined - some transactions have Amount=0
    df["Amount_log"] = np.log1p(df["Amount"])

    # Drop raw Time, amount - confirmed near-zero separability from Mann-Whitney
    df = df.drop(columns=["Time", "Amount"])

    logger.info(
        f"Engineered features added: Hour, Day, Amount_log | "
        f"Time dropped | shape: {df.shape}"
    )
    return df
