"""DataModule bundling the three split loaders and the shared statistics."""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.low_carbon_dataset import LowCarbonWindowDataset


class LowCarbonDataModule:
    def __init__(
        self,
        artifacts_dir: str = "artifacts/data/low_carbon",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        test_stride: int = 1,
    ):
        if not os.path.exists(os.path.join(artifacts_dir, "preprocess_stats.json")):
            raise FileNotFoundError(
                f"{artifacts_dir} not prepared — run src.data.low_carbon_preprocess first"
            )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # test_stride > 1 subsamples test forecast origins (e.g. 24 = one
        # non-overlapping forecast per day). It MUST come from the shared
        # common config so every model evaluates identical windows — the
        # prediction-contract alignment check enforces this.
        self.test_stride = int(test_stride)
        self.train_set = LowCarbonWindowDataset(artifacts_dir, "train")
        self.val_set = LowCarbonWindowDataset(artifacts_dir, "val")
        self.test_set = LowCarbonWindowDataset(artifacts_dir, "test", stride=self.test_stride)
        self.stats = self.train_set.stats
        self.target_transform = self.train_set.target_transform

    @property
    def gate_pos_weight(self) -> torch.Tensor:
        return torch.tensor(self.stats["gate_pos_weight"], dtype=torch.float32)

    @property
    def train_scale_raw(self) -> np.ndarray:
        return np.asarray(self.stats["train_scale_raw"], dtype=np.float64)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def train_loader(self):
        return self._loader(self.train_set, True)

    def val_loader(self):
        return self._loader(self.val_set, False)

    def test_loader(self):
        return self._loader(self.test_set, False)
