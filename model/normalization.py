import json
from pathlib import Path

import pandas as pd


def fit_and_save_scaler(train_df: pd.DataFrame, columns: list[str], save_path: Path) -> None:
    """
    Calculate the mean and standard deviation for specified columns and save to JSON.

    Zero standard deviations are replaced with 1.0 to prevent division by zero during normalization.

    Parameters
    ----------
    train_df : pd.DataFrame
        The training dataset.
    columns : list of str
        The list of columns to calculate statistics for.
    save_path : Path
        The file path where the calculated parameters will be saved as JSON.
    """

    means = train_df[columns].mean()
    stds = train_df[columns].std()
    stds[stds == 0.0] = 1.0

    means_dict = means.to_dict()
    stds_dict = stds.to_dict()

    data = {"mean": means_dict, "std": stds_dict}

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def normalize(df: pd.DataFrame, columns: list[str], scaler_path: Path) -> pd.DataFrame:
    """
    Normalize the specified columns in a DataFrame using saved mean and standard deviation.

    Returns a new copied DataFrame without modifying the input in-place. Zero standard
    deviations in the loaded parameters are replaced with 1.0.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to normalize.
    columns : list of str
        The list of columns to normalize.
    scaler_path : Path
        The file path to the JSON file containing the mean and standard deviation.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with normalized columns.
    """

    df = df.copy()
    with scaler_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    means = pd.Series(data["mean"])
    stds = pd.Series(data["std"])

    # Check for missing columns
    missing_cols = [col for col in columns if col not in means or col not in stds]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} are absent in the scaler parameters.")

    stds[stds == 0.0] = 1.0

    df[columns] = (df[columns] - means) / stds
    return df
