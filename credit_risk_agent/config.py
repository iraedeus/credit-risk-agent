"""
Configuration settings and constants for Credit Risk Intelligence System.

Defines filesystem paths, database column mappings, hyperparameter defaults,
and MLflow Model Registry alias constants.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Загружает переменные из .env в os.environ

__ROOT__ = Path(__file__).parent.parent
DATA_PATH = __ROOT__ / "data"
RAW_DATABASE_PATH = DATA_PATH / "database.db"
TRAIN_DATABASE_PATH = DATA_PATH / "train_database.db"
TEST_DATABASE_PATH = DATA_PATH / "test_database.db"

CLIENT_COLUMNS = ["client_id", "limit_bal", "sex", "education", "marriage", "age", "default"]
HISTORY_COLUMNS = ["client_id", "month", "pay_status", "bill_amt", "pay_amt"]
ID_COL = "client_id"
TARGET_COL = "default"
SCALER_COLS = ["pay_amt", "bill_amt", "limit_bal"]

ARTIFACTS_PATH = __ROOT__ / "artifacts"
ARTIFACTS_PATH.mkdir(exist_ok=True)

SCALER_PATH = ARTIFACTS_PATH / "scaler.json"
MODEL_SAVE_PATH = ARTIFACTS_PATH / "model.pt"

# Model Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 25
HIDDEN_SIZE = 64
DROPOUT_PROB = 0.28

BEST_MODEL_NAME = "CreditRiskModel"
BEST_MODEL_ALIAS = "champion"

# Microservices settings

_DATA_SERVICE_HOST = os.getenv("DATA_SERVICE_URL", "http://localhost")
_DATA_SERVICE_PORT = os.getenv("DATA_SERVICE_PORT", "8000")
DATA_SERVICE_URL = f"{_DATA_SERVICE_HOST}:{_DATA_SERVICE_PORT}"
