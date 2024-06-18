from sklearn.utils import Bunch
from torch_geometric.data import Data

from etflow.commons import MoleculeFeaturizer
from torch_geometric.data import Dataset
from .geom import GEOM

DATASET_MAPPING = {"geom": GEOM}


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
        use_ogb_feat: bool = False,
        use_edge_feat: bool = False,
    ):
        super().__init__()
        # instantiate dataset
        self.dataset_name = dataset_name
        self.dataset = DATASET_MAPPING[dataset_name]()
        self.mol_feat = MoleculeFeaturizer(use_ogb_feat=use_ogb_feat, use_edge_feat=use_edge_feat)
        self.cache = {}

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data_bunch: Bunch = self.dataset[idx]

        # get positions, atomic_numbers and smiles
        atomic_numbers = data_bunch["atomic_numbers"]
        pos = data_bunch["pos"]
        smiles = data_bunch["smiles"]

        # featurize molecule
        node_attr = self.mol_feat.get_atom_features(smiles)
        chiral_index, chiral_nbr_index, chiral_tag = self.mol_feat.get_chiral_centers(smiles)
        edge_index, edge_attr = self.mol_feat.get_edge_index(smiles)
        mol = self.mol_feat.get_mol_with_conformer(smiles, pos)

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
