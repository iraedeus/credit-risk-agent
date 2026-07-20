import numpy as np
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
