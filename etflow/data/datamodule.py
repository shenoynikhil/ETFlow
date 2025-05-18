"""Base Datamodule class using LightningDataModule
Takes as input a torch Dataset class, performs splitting
and returns dataloaders for train, val and test.
"""

from pathlib import Path
from typing import Dict

import lightning.pytorch as pl
from torch_geometric.loader import DataLoader

from .dataset import EuclideanDataset


class BaseDataModule(pl.LightningDataModule):
    """Datamodule to do all data stuff."""

    def __init__(
        self,
        data_dir: Path | None = None,
        partition: str = "drugs",
        dataloader_args: Dict = {},
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.partition = partition
        self.dataloader_args = dataloader_args

    def __repr__(self) -> str:
        return f"DataModule(partition={self.partition})"

    def setup(self, stage: str = None):
        """Prepares data splits for dataloader"""
        # Create train and val datasets for the specified partition
        self.train_dataset = EuclideanDataset(
            self.data_dir, partition=self.partition, split="train"
        )
        self.val_dataset = EuclideanDataset(
            self.data_dir, partition=self.partition, split="val"
        )

    def train_dataloader(self):
        """Creates train dataloader"""
        return DataLoader(self.train_dataset, **self.dataloader_args)

    def val_dataloader(self):
        """Creates val dataloader"""
        return DataLoader(self.val_dataset, **self.dataloader_args)

    def test_dataloader(self):
        """Creates test dataloader"""
        # Create test dataset for the specified partition
        self.test_dataset = EuclideanDataset(
            data_dir=self.data_dir,
            partition=self.partition,
            split="test",
        )

        return DataLoader(self.test_dataset, **self.dataloader_args)
