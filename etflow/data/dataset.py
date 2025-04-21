from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from etflow.commons.featurization import MoleculeFeaturizer
from etflow.commons.io import get_base_data_dir


class EuclideanDataset(Dataset):
    """Returns 3D Graph for different datasets

    Usage
    -----
    ```python
    from etflow.data import EuclideanDataset
    # pass path to processed data_dir
    dataset = EuclideanDataset(
        data_dir="processed",
        split="train",  # "train", "val", or "test"
        partition="drugs",  # "drugs" or "qm9"
    )
    ```
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        split: str = "train",
        partition: str = "drugs",
    ):
        super().__init__()
        self.mol_feat = MoleculeFeaturizer()

        # Set up paths
        if data_dir is None:
            self.data_dir = Path(get_base_data_dir()) / "processed"
        else:
            self.data_dir = Path(data_dir)

        # Set split and partition
        self.split = split
        self.partition = partition

        # Find all data files for the specified partition and split
        self.data_files = list((self.data_dir / partition.lower() / split).glob("*.pt"))

        if len(self.data_files) == 0:
            raise ValueError(
                f"No data files found for partition {partition} and split {split}"
            )

        # Sort files for reproducibility
        self.data_files.sort()

    def len(self):
        return len(self.data_files)

    def get(self, idx):
        # Load the data file
        data_path = self.data_files[idx]
        data = torch.load(data_path)

        # Get the molecule data
        smiles = data.smiles
        pos_confs = data.pos
        atomic_numbers = data.atomic_numbers

        # sample a random conformer
        conf_idx = np.random.randint(0, pos_confs.shape[0])
        pos = pos_confs[conf_idx]

        # Featurize molecule
        node_attr = self.mol_feat.get_atom_features(smiles)
        chiral_index, chiral_nbr_index, chiral_tag = self.mol_feat.get_chiral_centers(
            smiles
        )
        edge_index, edge_attr = self.mol_feat.get_edge_index(smiles, False)
        mol = self.mol_feat.get_mol_with_conformer(smiles, pos)

        # Create a new graph with additional features
        return Data(
            pos=pos,
            atomic_numbers=atomic_numbers,
            smiles=smiles,
            edge_index=edge_index,
            chiral_index=chiral_index,
            chiral_nbr_index=chiral_nbr_index,
            chiral_tag=chiral_tag,
            mol=mol,
            node_attr=node_attr,
            edge_attr=edge_attr,
        )
