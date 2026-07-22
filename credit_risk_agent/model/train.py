import sqlite3
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from credit_risk_agent.config import (
    ARTIFACTS_PATH,
    BATCH_SIZE,
    DATABASE_PATH,
    ID_COL,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    SCALER_COLS,
    SCALER_PATH,
    TARGET_COL,
)
from credit_risk_agent.data import fit_and_save_scaler, normalize, preprocess
from credit_risk_agent.model import CreditDefaultPredictor, prepare_dataset


def load_and_preprocess_data() -> pd.DataFrame:
    with sqlite3.connect(DATABASE_PATH) as conn:
        client_df = pd.read_sql_query("SELECT * FROM clients", conn)
        history_df = pd.read_sql_query("SELECT * FROM payment_history", conn)

        df = pd.merge(client_df, history_df, on="client_id")
        df = preprocess(df)
        return df


def load_and_preprocess_test_data() -> pd.DataFrame:
    test_df = pd.read_csv(ARTIFACTS_PATH / "test_clients.csv")
    with sqlite3.connect(DATABASE_PATH) as conn:
        test_df.to_sql("temp_test_ids", conn, if_exists="replace", index=False)
        client_df = pd.read_sql_query(
            "SELECT * FROM clients WHERE client_id IN (SELECT client_id FROM temp_test_ids)", conn
        )
        history_df = pd.read_sql_query(
            "SELECT * FROM payment_history WHERE client_id IN (SELECT client_id FROM temp_test_ids)", conn
        )
        conn.execute("DROP TABLE temp_test_ids")

        df = pd.merge(client_df, history_df, on="client_id")
        df = preprocess(df)
        return df


def split_and_save_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    client_data = df[[ID_COL, TARGET_COL]].drop_duplicates()
    train_ids, test_ids = train_test_split(
        client_data[ID_COL], test_size=0.2, stratify=client_data[TARGET_COL], random_state=42
    )

    train_df = df[df[ID_COL].isin(train_ids)]
    test_df = df[df[ID_COL].isin(test_ids)]

    fit_and_save_scaler(train_df, SCALER_COLS, SCALER_PATH)
    train_df = normalize(train_df, SCALER_COLS, SCALER_PATH)
    test_df = normalize(test_df, SCALER_COLS, SCALER_PATH)

    test_ids.to_csv(ARTIFACTS_PATH / "test_clients.csv", index=False)

    return train_df, test_df


def train_model(
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    model_save_path: Path,
) -> nn.Module:
    model = CreditDefaultPredictor(hidden_size=64, num_layers=1, static_size=14, dropout_prob=0.28)

    pos_weight = torch.tensor([78.0 / 22.0])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(25):
        model.train()
        epoch_losses = []
        for seq_features, static_features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(seq_features, static_features)
            loss_val = loss_fn(outputs, labels)
            epoch_losses.append(loss_val.item())

            loss_val.backward()
            optimizer.step()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1:02d}/25 - Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_save_path)

    return model


def check_model_quality(
    model: nn.Module, test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> None:
    model.eval()
    all_preds: list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for seq_features, static_features, labels in test_loader:
            outputs = model(seq_features, static_features)
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.view(-1).tolist())
            all_targets.extend(labels.view(-1).tolist())

    roc_auc = roc_auc_score(all_targets, all_preds)
    print(f"Test ROC AUC: {roc_auc:.4f}\n")

    binary_preds = [1 if p >= 0.55 else 0 for p in all_preds]
    print("Classification Report:")
    print(classification_report(all_targets, binary_preds, target_names=["Non-default", "Default"]))


if __name__ == "__main__":
    df = load_and_preprocess_data()
    train_df, test_df = split_and_save_ids(df)

    train_dataset = prepare_dataset(train_df, id_col=ID_COL, target_col=TARGET_COL)
    test_dataset = prepare_dataset(test_df, id_col=ID_COL, target_col=TARGET_COL)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = train_model(train_loader, MODEL_SAVE_PATH)
    check_model_quality(model, test_loader)
