import os.path as osp
from typing import Dict

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from etflow.commons import get_base_data_dir, load_memmap, load_npz


class GEOM(Dataset):
    """
    GEOM Dataset

    To use this dataset, you need to have the pre-processed data.
    We provide scripts to run using the raw GEOM data you can download.
    To run pre-processing, refer to the scripts in `scripts/preprocess_geom.py`.

    Usage:
    ```python
    from etflow.data import GEOM
    dataset = GEOM(data_dir="path_to_preprocessed_data")
    ```

    Parameters
    ----------
    data_dir: str
        Relative Path to pre-processed data. If `DATA_DIR` env variable is set,
        the processed_files will be loaded from `DATA_DIR/data_dir`. Make sure
        the `DATA_DIR` is set to the correct path.
    """

    processed_file_names: Dict[str, str] = {
        "atomic_inputs": "atomic_inputs.memmap",
        "energy": "energy.memmap",
        "pos_index_range": "pos_index_range.memmap",
        "smiles": "smiles.npz",
        "subset": "subset.npz",
    }

    def __init__(self, data_dir: str = None):
        super().__init__()

        base_data_dir = get_base_data_dir()
        data_dir = osp.join(base_data_dir, data_dir)
        self._check_files_exists(data_dir)

        path = osp.join(data_dir, self.processed_file_names["atomic_inputs"])
        self.atomic_inputs = load_memmap(path, np.float32).reshape(-1, 5)

        path = osp.join(data_dir, self.processed_file_names["pos_index_range"])
        self.pos_index_range = load_memmap(path, np.int32).reshape(-1, 2)

        path = osp.join(data_dir, self.processed_file_names["energy"])
        self.energy = load_memmap(path, np.float32).reshape(-1, 1)

        path = osp.join(data_dir, self.processed_file_names["smiles"])
        self.smiles, self.smiles_inv_indices = load_npz(path)

        path = osp.join(data_dir, self.processed_file_names["subset"])
        self.subset, self.subset_inv_indices = load_npz(path)

    def _check_files_exists(self, data_dir: str) -> bool:
        for _, file_path in self.processed_file_names.items():
            if not osp.exists(osp.join(data_dir, file_path)):
                return False

        return True

    def len(self):
        return self.energy.shape[0]

    def get(self, idx):
        # atomic information
        pos_idx_start, pos_idx_end = self.pos_index_range[idx]
        atomic_inputs = np.array(
            self.atomic_inputs[pos_idx_start:pos_idx_end], dtype=np.float32
        )
        positions = torch.from_numpy(atomic_inputs[:, 2:])
        atomic_numbers = torch.from_numpy(atomic_inputs[:, 0]).long()
        charges = torch.from_numpy(atomic_inputs[:, 1]).long()

        # other
        smiles = self.smiles[self.smiles_inv_indices[idx]]
        subset = self.subset[self.subset_inv_indices[idx]]

        return Data(
            atomic_numbers=atomic_numbers,
            charges=charges,
            pos=positions,
            smiles=smiles,
            subset=subset,
        )
