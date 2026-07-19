import json
from pathlib import Path

import pandas as pd


def fit_and_save_scaler(train_df: pd.DataFrame, columns: list[str], save_path: Path) -> None:
    means = train_df[columns].mean()
    stds = train_df[columns].std()
    stds[stds == 0.0] = 1.0

    means_dict = means.to_dict()
    stds_dict = stds.to_dict()

    data = {"mean": means_dict, "std": stds_dict}

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def normalize(df: pd.DataFrame, columns: list[str], scaler_path: Path) -> pd.DataFrame:
    df = df.copy()
    with scaler_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    means = pd.Series(data["mean"])
    stds = pd.Series(data["std"])
    df[columns] = (df[columns] - means) / stds
    return df
