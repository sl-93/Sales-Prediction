from pathlib import Path
import pandas as pd
import yaml
import numpy as np
from typing import List
import jdatetime as jdt
import matplotlib.pyplot as plt
from src import logger
from sklearn.metrics import mean_squared_error, r2_score
import os



# ====== Sarima =====
def read_yaml(path_to_yaml: Path):
    with open(path_to_yaml) as yaml_file:
        config = yaml.safe_load(yaml_file)
        return config


def ingest_folder_files(path_to_folder: Path) -> List:
    try:
        folder_path = Path(path_to_folder)
        files = [f for f in folder_path.iterdir() if f.is_file()]
        logger.info("CSV files have been ingested successfully!")
    except Exception as e:
        logger.info(e)
    return files


def get_year_week(jdate: jdt.datetime) -> int:
        """Return week number of Jalali date (1-based)."""
        year_start = jdt.datetime(jdate.year, 1, 1)
        delta_days = (jdate.togregorian() - year_start.togregorian()).days
        return (delta_days // 7) + 1


def to_jdatetime(dates: pd.Series) -> pd.Series:
    """Convert YYYYMMDD integers to jdatetime objects."""
    return [jdt.datetime.strptime(str(x), "%Y%m%d") for x in dates]


def split_train_forecast(df, train_len):
    
    return (
        df.iloc[:train_len].reset_index(drop=True),
        df.iloc[train_len:].reset_index(drop=True)
    )

