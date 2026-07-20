import json
from pathlib import Path

import pandas as pd


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    mean : dict of str to float, optional
        Means for each feature.
    std : dict of str to float, optional
        Standard deviations for each feature.
    """

    def __init__(self, mean: dict[str, float] | None = None, std: dict[str, float] | None = None) -> None:
        self.mean = mean or {}
        self.std = std or {}

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "StandardScaler":
        """
        Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        df : pd.DataFrame
            The input training DataFrame.
        columns : list of str
            The list of columns to fit the statistics on.
        """
        means = df[columns].mean()
        stds = df[columns].std()
        stds[stds == 0.0] = 1.0

        self.mean = means.to_dict()
        self.std = stds.to_dict()
        return self

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Perform standardization by centering and scaling.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to normalize.
        columns : list of str
            The list of columns to normalize.
        """
        df = df.copy()
        means = pd.Series(self.mean)
        stds = pd.Series(self.std)

        # Check for missing columns
        missing_cols = [col for col in columns if col not in means.index or col not in stds.index]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} are absent in the scaler parameters.")

        stds[stds == 0.0] = 1.0
        df[columns] = (df[columns] - means) / stds
        return df

    def save(self, path: Path) -> None:
        """
        Save the scaler parameters to a JSON file.

        Parameters
        ----------
        path : Path
            The file path where parameters will be saved as JSON.
        """
        data = {"mean": self.mean, "std": self.std}
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "StandardScaler":
        """
        Load scaler parameters from a JSON file.

        Parameters
        ----------
        path : Path
            The file path to the JSON file containing the parameters.
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(mean=data["mean"], std=data["std"])


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
    scaler = StandardScaler().fit(train_df, columns)
    scaler.save(save_path)


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
    scaler = StandardScaler.load(scaler_path)
    return scaler.transform(df, columns)
