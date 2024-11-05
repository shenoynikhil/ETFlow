from torch_geometric.data import Data, Dataset

from etflow.commons.featurization import MoleculeFeaturizer

from .geom import GEOM


class EuclideanDataset(Dataset):
    """Returns 3D Graph for different datasets

    Usage
    -----
    ```python
    from etflow.data import EuclideanDataset
    # pass path to processed data_dir
    dataset = EuclideanDataset(data_dir=<>)
    ```
    """

    def __init__(
        self,
        data_dir: str,
        use_ogb_feat: bool = False,
        use_edge_feat: bool = False,
    ):
        super().__init__()
        # instantiate dataset
        self.dataset = GEOM(data_dir=data_dir)
        self.mol_feat = MoleculeFeaturizer()
        self.use_ogb_feat = use_ogb_feat
        self.use_edge_feat = use_edge_feat
        self.cache = {}

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data_bunch = self.dataset[idx]

        # get positions, atomic_numbers and smiles
        atomic_numbers = data_bunch["atomic_numbers"]
        pos = data_bunch["pos"]
        smiles = data_bunch["smiles"]

        # featurize molecule
        node_attr = self.mol_feat.get_atom_features(smiles, self.use_ogb_feat)
        chiral_index, chiral_nbr_index, chiral_tag = self.mol_feat.get_chiral_centers(
            smiles
        )
        edge_index, edge_attr = self.mol_feat.get_edge_index(smiles, self.use_edge_feat)
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
