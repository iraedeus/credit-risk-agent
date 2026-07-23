import argparse
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
    BATCH_SIZE,
    ID_COL,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    RAW_DATABASE_PATH,
    SCALER_COLS,
    SCALER_PATH,
    TARGET_COL,
    TEST_DATABASE_PATH,
    TRAIN_DATABASE_PATH,
)
from credit_risk_agent.data import StandardScaler, preprocess
from credit_risk_agent.model import CreditDefaultPredictor, prepare_dataset


def load_and_preprocess_from_db(db_path: Path) -> pd.DataFrame:
    """
    Load raw relational tables from a SQLite database, merge them, and apply preprocessing.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file containing `clients`, `payment_history`,
        and `ground_truth` tables.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame containing combined client features and payment history.
    """

    with sqlite3.connect(db_path) as conn:
        client_df = pd.read_sql_query("SELECT * FROM clients", conn)
        history_df = pd.read_sql_query("SELECT * FROM payment_history", conn)
        gt_df = pd.read_sql_query("SELECT * FROM ground_truth", conn)

        df = pd.merge(client_df, gt_df, on="client_id")
        df = pd.merge(df, history_df, on="client_id")
        df = preprocess(df)
        return df


def save_split_db() -> None:
    """
    Split raw database records into stratified training and testing SQLite databases.

    Reads client records from `RAW_DATABASE_PATH`, performs stratified train-test splitting
    on the ground truth target column, and saves the partitioned relational tables (`clients`,
    `payment_history`, `ground_truth`) into `TRAIN_DATABASE_PATH` and `TEST_DATABASE_PATH`.

    Returns
    -------
    None
    """

    with sqlite3.connect(RAW_DATABASE_PATH) as raw_conn:
        raw_clients = pd.read_sql_query("SELECT * FROM clients", raw_conn)
        raw_history = pd.read_sql_query("SELECT * FROM payment_history", raw_conn)
        raw_gt = pd.read_sql_query("SELECT * FROM ground_truth", raw_conn)

        train_ids, test_ids = train_test_split(
            raw_gt[ID_COL], test_size=0.2, stratify=raw_gt[TARGET_COL], random_state=42
        )

        train_clients = raw_clients[raw_clients["client_id"].isin(train_ids)]
        train_history = raw_history[raw_history["client_id"].isin(train_ids)]
        train_gt = raw_gt[raw_gt["client_id"].isin(train_ids)]

        test_clients = raw_clients[raw_clients["client_id"].isin(test_ids)]
        test_history = raw_history[raw_history["client_id"].isin(test_ids)]
        test_gt = raw_gt[raw_gt["client_id"].isin(test_ids)]

    with sqlite3.connect(TRAIN_DATABASE_PATH) as train_conn:
        train_clients.to_sql("clients", train_conn, if_exists="replace", index=False)
        train_history.to_sql("payment_history", train_conn, if_exists="replace", index=False)
        train_gt.to_sql("ground_truth", train_conn, if_exists="replace", index=False)

    with sqlite3.connect(TEST_DATABASE_PATH) as test_conn:
        test_clients.to_sql("clients", test_conn, if_exists="replace", index=False)
        test_history.to_sql("payment_history", test_conn, if_exists="replace", index=False)
        test_gt.to_sql("ground_truth", test_conn, if_exists="replace", index=False)


def train_model(
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    model_save_path: Path,
) -> nn.Module:
    """
    Train the CreditDefaultPredictor neural network model and save trained weights.

    Parameters
    ----------
    train_loader : DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        DataLoader yielding batches of sequence features, static features, and target labels.
    model_save_path : Path
        Destination filepath to save the trained model weights PyTorch checkpoint.

    Returns
    -------
    nn.Module
        Trained CreditDefaultPredictor model instance.
    """

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
    """
    Evaluate trained model performance on test dataset and print quality metrics.

    Computes ROC-AUC score and prints a classification report including precision,
    recall, and F1-score for default prediction.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model instance.
    test_loader : DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        DataLoader providing test features and target labels.

    Returns
    -------
    None
    """

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


def main() -> None:
    """
    Execute main pipeline for database splitting, model training, and evaluation.

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(description="Скрипт для загрузки данных и обучения модели")
    parser.add_argument(
        "--view-quality", action="store_true", help="Выдать отчёт качества обученной модели на тестовых данных"
    )
    args = parser.parse_args()

    view_quality = args.view_quality

    save_split_db()

    if view_quality:
        print("Загрузка сохраненной модели и оценка качества на тестовой выборке...")
        test_df = load_and_preprocess_from_db(TEST_DATABASE_PATH)
        scaler = StandardScaler.load(SCALER_PATH)
        test_df = scaler.transform(test_df, SCALER_COLS)

        test_dataset = prepare_dataset(test_df, id_col=ID_COL, target_col=TARGET_COL)
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = CreditDefaultPredictor(hidden_size=64, num_layers=1, static_size=14, dropout_prob=0.28)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        check_model_quality(model, test_loader)
        return

    train_df = load_and_preprocess_from_db(TRAIN_DATABASE_PATH)
    test_df = load_and_preprocess_from_db(TEST_DATABASE_PATH)

    scaler = StandardScaler().fit(train_df, SCALER_COLS)
    scaler.save(SCALER_PATH)
    train_df = scaler.transform(train_df, SCALER_COLS)
    test_df = scaler.transform(test_df, SCALER_COLS)

    train_dataset = prepare_dataset(train_df, id_col=ID_COL, target_col=TARGET_COL)
    test_dataset = prepare_dataset(test_df, id_col=ID_COL, target_col=TARGET_COL)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = train_model(train_loader, MODEL_SAVE_PATH)
    check_model_quality(model, test_loader)


if __name__ == "__main__":
    main()
