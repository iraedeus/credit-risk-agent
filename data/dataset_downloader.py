import kaggle
import sqlite3
import pandas as pd
from pathlib import Path

__ROOT__ = Path(__file__).parent.parent
DATASET_PATH = __ROOT__ / "data"

kaggle.api.authenticate()
kaggle.api.dataset_download_files('uciml/default-of-credit-card-clients-dataset', path=DATASET_PATH, unzip=True)

df = pd.read_csv(DATASET_PATH / "UCI_Credit_Card.csv")
print(df.head())
