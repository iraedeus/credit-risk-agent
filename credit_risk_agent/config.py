from pathlib import Path

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
