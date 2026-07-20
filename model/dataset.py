import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from model.preprocessing import preprocess_static


class CreditDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    PyTorch Dataset wrapper for Credit Risk data.

    This dataset yields a tuple containing:
    1. A 3D sequence tensor of shape (seq_len, 3) representing payment history.
    2. A 1D static tensor of shape (14,) representing client demographic and limit characteristics.
    3. A 1D target tensor of shape (1,) representing the default label (0 or 1).
    """

    def __init__(self, sequences: np.ndarray, static_features: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.static_features = torch.tensor(static_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.static_features[idx], self.labels[idx]


def prepare_dataset(
    df: pd.DataFrame,
    id_col: str = "client_id",
    target_col: str = "default",
) -> CreditDataset:
    """
    Pivot sequence history, extract and encode static characteristics, and bundle
    them into a CreditDataset.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame containing client history and characteristics.
    id_col : str, default='client_id'
        Column name identifying clients.
    target_col : str, default='default'
        Column name for default targets.

    Returns
    -------
    CreditDataset
        Dataset ready for PyTorch model training or evaluation.
    """
    # Check sequence lengths: each client must have exactly 6 records
    counts = df.groupby(id_col).size()
    invalid_clients = counts[counts != 6]
    if not invalid_clients.empty:
        raise ValueError(f"Clients {invalid_clients.index.tolist()} must have exactly 6 records.")

    # 1. Pivot sequence history into a 3D array (batch_size, 6, 3)
    df_wide = df.pivot(index=id_col, columns="month", values=["bill_amt", "pay_amt", "pay_status"])
    df_wide = df_wide.reorder_levels([1, 0], axis=1).sort_index(axis=1)
    features_3d = df_wide.values.reshape(-1, 6, 3)

    # 2. Extract static characteristics (one row per client) and encode them
    df_static = df.groupby(id_col).first().reset_index()
    static_features = preprocess_static(df_static).values

    # 3. Extract targets (one label per client)
    labels = df.groupby(id_col)[target_col].first().values

    return CreditDataset(features_3d, static_features, labels)
