"""
Model training CLI script, evaluation pipeline, and MLflow champion registration.
"""

import argparse
import copy
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from sklearn.metrics import classification_report, roc_auc_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from credit_risk_agent.config import (
    BATCH_SIZE,
    BEST_MODEL_ALIAS,
    BEST_MODEL_NAME,
    DROPOUT_PROB,
    EPOCHS,
    HIDDEN_SIZE,
    ID_COL,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    SCALER_COLS,
    SCALER_PATH,
    TARGET_COL,
    TEST_DATABASE_PATH,
    TRAIN_DATABASE_PATH,
)
from credit_risk_agent.data import StandardScaler
from credit_risk_agent.data.loader import load_and_preprocess_from_db
from credit_risk_agent.model import CreditDefaultModel, prepare_dataset
from credit_risk_agent.model.loader import (
    load_model_from_registry,
    load_model_from_run,
    load_scaler_from_registry,
    load_scaler_from_run,
)


def configure_argparser() -> argparse.Namespace:
    """
    Parse command-line arguments for training and model evaluation CLI.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments containing hyperparameters and evaluation flags.
    """
    parser = argparse.ArgumentParser(description="Скрипт для загрузки данных и обучения модели")
    parser.add_argument(
        "--view-quality", action="store_true", help="Выдать отчёт качества обученной модели на тестовых данных"
    )
    parser.add_argument("--run-id", type=str, help="ID запуска в MLflow для загрузки модели и скейлера")

    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Количество эпох обучения")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Установить параметр learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Установить параметр batch size")
    parser.add_argument("--hidden", type=int, default=HIDDEN_SIZE, help="Установить параметр hidden_layers")
    parser.add_argument("--run-name", type=str, default="baseline_run", help="Имя запуска в MLflow")

    return parser.parse_args()


def train_model(
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    model_save_path: Path,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = 1,
    dropout_prob: float = DROPOUT_PROB,
) -> tuple[nn.Module, float]:
    """
    Train the CreditDefaultModel neural network model and save trained weights.

    Parameters
    ----------
    train_loader : DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        DataLoader yielding batches of sequence features, static features, and target labels.
    model_save_path : Path
        Destination filepath to save the trained model weights PyTorch checkpoint.
    epochs : int, default=25
        Number of training epochs.
    lr : float, default=LEARNING_RATE
        Learning rate for Adam optimizer.
    hidden_size : int, default=64
        Hidden state dimension of LSTM model.
    num_layers : int, default=1
        Number of LSTM layers.
    dropout_prob : float, default=0.28
        Dropout probability.

    Returns
    -------
    tuple[nn.Module, float]
        Tuple containing the trained CreditDefaultModel instance and the best loss value achieved.
    """

    model = CreditDefaultModel(
        hidden_size=hidden_size, num_layers=num_layers, static_size=14, dropout_prob=dropout_prob
    )

    pos_weight = torch.tensor([78.0 / 22.0])  # Classes ratio
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_weights = None

    print("Начинаем обучение...")
    for epoch in range(epochs):
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

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = copy.deepcopy(model.state_dict())

        print(f"Эпоха {epoch + 1:02d}/{epochs} - Средний лосс: {avg_loss:.4f}")
        if mlflow.active_run():
            mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)

    if mlflow.active_run():
        mlflow.log_metric("best_train_loss", best_loss)

    if best_weights is not None:
        torch.save(best_weights, model_save_path)

    model.load_state_dict(best_weights)
    return model, best_loss


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

    roc_auc = float(roc_auc_score(all_targets, all_preds))
    print(f"Тестовый ROC AUC: {roc_auc:.4f}\n")

    binary_preds = [1 if p >= 0.55 else 0 for p in all_preds]
    report = classification_report(all_targets, binary_preds, target_names=["Non-default", "Default"], output_dict=True)
    print("Отчет классификации:")
    print(classification_report(all_targets, binary_preds, target_names=["Non-default", "Default"]))

    metrics = {
        "test_roc_auc": roc_auc,
        "default_precision": float(report["Default"]["precision"]),
        "default_recall": float(report["Default"]["recall"]),
        "default_f1": float(report["Default"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
    }
    if mlflow.active_run():
        mlflow.log_metrics(metrics)


def save_champion_model(loss: float) -> None:
    """
    Compare current model loss against the champion in MLflow Model Registry
    and update champion alias if record is broken.

    Parameters
    ----------
    loss : float
        Best training loss achieved by the current model.

    Returns
    -------
    None
    """
    client = mlflow.MlflowClient()
    current_run_id = mlflow.active_run().info.run_id

    try:
        champion = client.get_model_version_by_alias(BEST_MODEL_NAME, BEST_MODEL_ALIAS)
        champion_run = client.get_run(champion.run_id)
        champion_loss = champion_run.data.metrics.get("best_train_loss") or float("inf")
    except mlflow.exceptions.MlflowException:
        champion_loss = float("inf")
        print("Чемпион еще не назначен.")

    if loss < champion_loss:
        print(f"Новый рекорд! Лосс улучшился с {champion_loss:.4f} до {loss:.4f}")
        mv = mlflow.register_model(f"runs:/{current_run_id}/model", BEST_MODEL_NAME)
        client.set_registered_model_alias(BEST_MODEL_NAME, BEST_MODEL_ALIAS, mv.version)
    else:
        print(f"Модель не побила рекорд. Текущий лосс чемпиона: {champion_loss:.4f}, наш лосс: {loss:.4f}")


def main() -> None:
    """
    Execute main pipeline for database splitting, model training, and evaluation.

    Returns
    -------
    None
    """

    args = configure_argparser()

    if args.view_quality:
        if not TEST_DATABASE_PATH.exists():
            raise FileNotFoundError("Тестовая БД не существует. Пожалуйста запустите скрипт подготовки данных.")

        print("Загрузка сохраненной модели и оценка качества на тестовой выборке...")
        test_df = load_and_preprocess_from_db(TEST_DATABASE_PATH)
        if args.run_id:
            scaler = load_scaler_from_run(args.run_id)
            model = load_model_from_run(args.run_id)
        else:
            scaler = load_scaler_from_registry(BEST_MODEL_NAME, BEST_MODEL_ALIAS)
            model = load_model_from_registry(BEST_MODEL_NAME, BEST_MODEL_ALIAS)
        test_df = scaler.transform(test_df, SCALER_COLS)

        test_dataset = prepare_dataset(test_df, id_col=ID_COL, target_col=TARGET_COL)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

        check_model_quality(model, test_loader)
        return

    if not TRAIN_DATABASE_PATH.exists() or not TEST_DATABASE_PATH.exists():
        raise FileNotFoundError(
            "Тренировочная или тестовая БД не существует. Пожалуйста запустите скрипт подготовки данных."
        )

    train_df = load_and_preprocess_from_db(TRAIN_DATABASE_PATH)
    test_df = load_and_preprocess_from_db(TEST_DATABASE_PATH)

    scaler = StandardScaler.load(SCALER_PATH)
    train_df = scaler.transform(train_df, SCALER_COLS)
    test_df = scaler.transform(test_df, SCALER_COLS)

    train_dataset = prepare_dataset(train_df, id_col=ID_COL, target_col=TARGET_COL)
    test_dataset = prepare_dataset(test_df, id_col=ID_COL, target_col=TARGET_COL)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    mlflow.set_experiment("credit_default_predictor")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "hidden_size": args.hidden,
                "epochs": args.epochs,
                "dropout_prob": DROPOUT_PROB,
                "optimizer": "Adam",
            }
        )

        mlflow.log_artifact(str(SCALER_PATH), artifact_path="preprocessing")

        model, loss = train_model(
            train_loader, MODEL_SAVE_PATH, epochs=args.epochs, lr=args.lr, hidden_size=args.hidden
        )
        check_model_quality(model, test_loader)

        if isinstance(model, torch.nn.Module):
            mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")
            save_champion_model(loss)


if __name__ == "__main__":
    main()
