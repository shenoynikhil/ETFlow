import datamol as dm
import torch
from sklearn.utils import Bunch
from torch_geometric.data import Data
from datamol.types import Mol

from etflow.commons import (
    atom_to_feature_vector,
    build_conformer,
    compute_edge_index,
    get_chiral_tensors,
)
from torch_geometric.data import Dataset
from .geom import GEOM

DATASET_MAPPING = {
    "geom": GEOM,
}


class EuclideanDataset(Dataset):
    """Returns 3D Graph for different datasets

    Usage:

    ```python
    from etflow.data import EuclideanDataset

    dataset = EuclideanDataset("geom")

    # with node features passed
    dataset = EuclideanDataset("geom", with_node_feat=True)

    # with edge features passed
    dataset = EuclideanDataset("geom", with_edge_feat=True)
    ```

    """

    def __init__(
        self,
        dataset_name: str,
        with_atom_feat: bool = False,
        with_bond_feat: bool = False,
    ):
        super().__init__()
        # instantiate dataset
        self.dataset_name = dataset_name
        self.dataset = DATASET_MAPPING[dataset_name]()
        self.with_atom_feat = with_atom_feat
        self.with_bond_feat = with_bond_feat
        self.cache = {}

    def get(self, idx):
        data_bunch: Bunch = self.dataset[idx]

        # get positions, atomic_numbers and smiles
        atomic_numbers = data_bunch["atomic_numbers"]
        pos = data_bunch["pos"]
        smiles = data_bunch["smiles"]
        charges = data_bunch["charges"]
        mol = dm.to_mol(smiles, remove_hs=False, ordered=True)

        if smiles in self.cache:
            (
                node_attr,
                chiral_index,
                chiral_nbr_index,
                chiral_tag,
                edge_attr,
                edge_index
            ) = self.cache[smiles]
        else:
            # chirality stuff
            edge_index, edge_attr = compute_edge_index(
                mol, with_edge_attr=self.use_edge_feat
            )
            chiral_index, chiral_nbr_index, chiral_tag = get_chiral_tensors(mol)
            node_attr = self.compute_node_attr(
                smiles=smiles,
                mol=mol,
                charges=charges,
                edge_index=edge_index,
                num_nodes=pos.shape[0],
            )
            self.cache[smiles] = (
                node_attr,
                chiral_index,
                chiral_nbr_index,
                chiral_tag,
                edge_attr,
                edge_index,
            )

        # update mol
        mol.AddConformer(build_conformer(pos))

        graph = Data(
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

        return graph

    def compute_node_attr(self, mol: Mol, charges) -> torch.Tensor:
        '''
        Node attributes set to charges if `with_atom_feat` is `False`
        else set to a 10 dimensional feature vector. 
        Check `etflow.commons.atom_to_feature_vector` for more details.
        '''
        # compute node features if `with_atom_feat` is `True`
        node_attr = torch.tensor(
            [atom_to_feature_vector(atom) for atom in mol.GetAtoms()],
            dtype=torch.float32,
        ) if self.with_atom_feat else charges.view(-1, 1).float()

        return node_attr.float()
