import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CreditDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences)
        self.labels = torch.tensor(labels).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def prepare_dataset(
    df: pd.DataFrame,
    id_col: str = "client_id",
    target_col: str = "default",
) -> CreditDataset:
    """
    Pivot DataFrame, reshape features into a 3D sequence array, extract labels,
    and wrap them inside a CreditDataset.

    Parameters
    ----------
    df : pd.DataFrame
        The input credit history DataFrame.
    id_col : str, default='client_id'
        Column name identifying clients.
    target_col : str, default='default'
        Column name for default targets.

    Returns
    -------
    CreditDataset
        Ready-to-use PyTorch dataset.
    """
    df_wide = df.pivot(index=id_col, columns="month", values=["bill_amt", "pay_amt", "pay_status"])
    df_wide = df_wide.reorder_levels([1, 0], axis=1).sort_index(axis=1)
    features_3d = df_wide.values.reshape(-1, 6, 3)
    labels = df.groupby(id_col)[target_col].first().values
    return CreditDataset(features_3d, labels)
