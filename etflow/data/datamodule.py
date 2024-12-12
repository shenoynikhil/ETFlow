"""Base Datamodule class using LightningDataModule
Takes as input a torch Dataset class, performs splitting
and returns dataloaders for train, val and test.
"""

import os
import random
from typing import Optional

import lightning.pytorch as pl
import numpy as np
from loguru import logger as log
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader

from etflow.commons.io import get_base_data_dir


class BaseDataModule(pl.LightningDataModule):
    """Datamodule to do all data stuff."""

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        dataloader_args: dict = {},
        train_indices_path: Optional[str] = None,
        val_indices_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.dataloader_args = dataloader_args

        self.train_indices_path = train_indices_path
        self.val_indices_path = val_indices_path

    def __repr__(self) -> str:
        return f"DataModule(dataset={self.dataset})"

    def setup(self, stage: str = None):
        """Prepares data splits for dataloader
        By default, splits data randomly into train, val and test, 80:10:10
        """
        # perform splits if train, val and test datasets are not passed
        if self.train_indices_path is not None and self.val_indices_path is not None:
            # update indices path by appending base data dir
            base_data_dir = get_base_data_dir()
            self.train_indices_path = os.path.join(
                base_data_dir, self.train_indices_path
            )
            self.val_indices_path = os.path.join(base_data_dir, self.val_indices_path)

            self.train_indices = np.load(self.train_indices_path, allow_pickle=True)
            self.val_indices = np.load(self.val_indices_path, allow_pickle=True)
        else:
            all_indices = list(range(len(self.dataset)))

            log.info("Performing train, val, test split")
            random.shuffle(all_indices)  # inplace shuffle
            self.train_indices, self.val_indices = (
                all_indices[: int(len(all_indices) * 0.8)],
                all_indices[int(len(all_indices) * 0.8) : int(len(all_indices) * 0.9)],
            )

        # create datasets
        self.train_dataset = Subset(self.dataset, self.train_indices)
        self.val_dataset = Subset(self.dataset, self.val_indices)

    def train_dataloader(self):
        """Creates train dataloader"""
        return DataLoader(self.train_dataset, **self.dataloader_args)

    def val_dataloader(self):
        """Creates val dataloader"""
        return DataLoader(self.val_dataset, **self.dataloader_args)
