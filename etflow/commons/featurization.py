# allowable multiple choice node and edge features
from copy import deepcopy
from typing import Tuple

import datamol as dm
import numpy as np
import torch
from datamol.types import Mol
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
from torch_cluster import radius_graph
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_sparse import coalesce

from .covmat import build_conformer

# similar to GeoMol
BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
chirality = {
    ChiralType.CHI_TETRAHEDRAL_CW: -1.0,
    ChiralType.CHI_TETRAHEDRAL_CCW: 1.0,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0,
}

allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "misc",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_implicit_valence": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}


class MoleculeFeaturizer:
    """A Featurizer Class for Molecules.
    - Give smiles, get mol objects, atom features, bond features, etc.
    - Caching to avoid recomputation.

    Parameters
    ----------
    use_ogb_features: bool, default=True
        If True, 10-dimensional atom features based on OGB are computed,
        Otherwise, atomic charges are used.
    use_edge_feat: bool, default=False
        If True, edge features are computed.
    """

    def __init__(self, use_ogb_feat: bool = True, use_edge_feat: bool = False):
        # smiles based cache
        self.cache = {}
        self.use_ogb_feat = use_ogb_feat
        self.use_edge_feat = use_edge_feat

    def get_mol(self, smiles: str) -> Mol:
        return dm.to_mol(smiles, remove_hs=False, ordered=True)

    def get_atom_features(self, smiles: str) -> torch.Tensor:
        # check if cached
        if smiles in self.cache and "atom_features" in self.cache[smiles]:
            return self.cache[smiles]["atom_features"]

        # compute atom features
        mol = self.get_mol(smiles)

        if self.use_ogb_feat:
            atom_features = torch.tensor(
                [atom_to_feature_vector(atom) for atom in mol.GetAtoms()],
                dtype=torch.float32,
            )  # (n_atoms, 10)
        else:
            atom_features = torch.tensor(
                [atom.GetFormalCharge() for atom in mol.GetAtoms()],
                dtype=torch.float32,
            ).view(
                -1, 1
            )  # (n_atoms, 1)

        # add smiles to cache
        if smiles not in self.cache:
            self.cache[smiles] = {}

        self.cache[smiles]["atom_features"] = atom_features
        return atom_features

    def get_chiral_centers(self, smiles: str) -> torch.Tensor:
        # check if cached
        if smiles in self.cache and "chiral_centers" in self.cache[smiles]:
            return self.cache[smiles]["chiral_centers"]

        # compute chiral centers
        mol = self.get_mol(smiles)
        chiral_index, chiral_nbr_index, chiral_tag = get_chiral_tensors(mol)

        # add smiles to cache
        if smiles not in self.cache:
            self.cache[smiles] = {}

        self.cache[smiles]["chiral_centers"] = (
            chiral_index,
            chiral_nbr_index,
            chiral_tag,
        )
        return chiral_index, chiral_nbr_index, chiral_tag

    def get_mol_with_conformer(self, smiles: str, positions: torch.Tensor) -> Mol:
        mol = self.get_mol(smiles)
        mol.AddConformer(build_conformer(positions))
        return mol

    def get_edge_index(self, smiles: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns edge index and edge attributes for a given smiles."""
        # check if cached
        if smiles in self.cache and "edge_index" in self.cache[smiles]:
            return self.cache[smiles]["edge_index"], self.cache[smiles]["edge_attr"]

        # compute edge index
        mol = self.get_mol(smiles)
        edge_index, edge_attr = compute_edge_index(
            mol, with_edge_attr=self.use_edge_feat
        )

        # add smiles to cache
        if smiles not in self.cache:
            self.cache[smiles] = {}

        self.cache[smiles]["edge_index"] = edge_index
        self.cache[smiles]["edge_attr"] = edge_attr
        return edge_index, edge_attr


def get_atomic_number_and_charge(mol: Chem.Mol):
    """Returns atoms number and charge for rdkit molecule"""
    return np.array(
        [[atom.GetAtomicNum(), atom.GetFormalCharge()] for atom in mol.GetAtoms()]
    )


def GetNumRings(atom):
    return sum([atom.IsInRingSize(i) for i in range(3, 7)])


def atom_to_feature_vector(atom):
    """Node Invariant Features for an Atom."""
    atom_feature = [
        safe_index(
            allowable_features["possible_chirality_list"],
            chirality[atom.GetChiralTag()],
        ),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(
            allowable_features["possible_implicit_valence"], atom.GetImplicitValence()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
        GetNumRings(atom),
    ]
    return atom_feature


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except Exception as e:
        return len(l) - 1


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        # allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
        # allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def compute_edge_index(
    mol, no_reverse: bool = False, with_edge_attr=False
) -> torch.Tensor:
    """Computes edge index from mol object"""
    edge_list = []
    bond_types = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_list.append((i, j))
        bond_types.append(bond_to_feature_vector(bond))

        if not no_reverse:
            edge_list.append((j, i))
            bond_types.append(bond_to_feature_vector(bond))

    if len(edge_list) == 0:
        return torch.empty((2, 0)).long()

    edge_index = torch.from_numpy(np.array(edge_list).T).long()

    if with_edge_attr:
        edge_attr = torch.tensor(bond_types, dtype=torch.float32)  # (num_edges, 1)
        return edge_index, edge_attr

    return edge_index, None


def _extend_graph_order(
    num_nodes: int, edge_index: torch.Tensor, edge_type: torch.Tensor, order=3
):
    """
    Extends order of the existing bond index.

    For instance, if atom-1-atom-2-atom-3 form an bond angle, then atom-1-atom-3
    will be added to the bond index for order=3.

    The importance of this is highlighted in section 2.1 of the paper:
    https://arxiv.org/abs/1909.11459

    Parameters
    ----------
    num_nodes: int
        Number of atoms.
    edge_index: torch.Tensor
        Bond indices of the original graph.
    edge_type: torch.Tensor
        Bond types of the original graph.
    order: int
        Extension order.

    Returns
    -------
    new_edge_index: torch.Tensor
        Extended edge indices.
    new_edge_type: torch.Tensor
        Extended edge types.
    """

    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        Returns:
            Following attributes will be updated:
              - edge_index
              - edge_type
            Following attributes will be added to the data object:
              - bond_edge_index:  Original edge_index.
        """
        adj_mats = [
            torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
            binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device)),
        ]

        for i in range(2, order + 1):
            adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

        return order_mat

    num_types = len(BOND_TYPES)

    N = num_nodes
    adj = to_dense_adj(edge_index).squeeze(0)
    adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

    type_mat = to_dense_adj(edge_index, edge_attr=edge_type).squeeze(0)  # (N, N)
    type_highorder = torch.where(
        adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order)
    )
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder

    new_edge_index, new_edge_type = dense_to_sparse(type_new)

    # data.bond_edge_index = data.edge_index  # Save original edges
    new_edge_index, new_edge_type = coalesce(
        new_edge_index, new_edge_type.long(), N, N
    )  # modify data

    return new_edge_index, new_edge_type


def _extend_to_radius_graph(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float = 10.0,
    max_num_neighbors: int = 32,
    unspecified_type_number=0,
):
    assert edge_type.dim() == 1
    N = pos.size(0)

    bgraph_adj = torch.sparse.LongTensor(edge_index, edge_type, torch.Size([N, N]))
    rgraph_edge_index = radius_graph(
        pos, r=cutoff, batch=batch, max_num_neighbors=max_num_neighbors
    )  # (2, E_r)

    rgraph_adj = torch.sparse.LongTensor(
        rgraph_edge_index,
        torch.ones(rgraph_edge_index.size(1)).long().to(pos.device)
        * unspecified_type_number,
        torch.Size([N, N]),
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)

    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()

    return new_edge_index, new_edge_type


def extend_graph_order_radius(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float = 10.0,
    order: int = 3,
    max_num_neighbors: int = 32,
    extend_radius: bool = True,
    extend_order: bool = True,
):
    """Extends bond index"""
    num_nodes = pos.shape[0]
    if extend_order:
        edge_index, edge_type = _extend_graph_order(
            num_nodes=num_nodes, edge_index=edge_index, edge_type=edge_type, order=order
        )

    if extend_radius:
        edge_index, edge_type = _extend_to_radius_graph(
            pos=pos,
            edge_index=edge_index,
            edge_type=edge_type,
            cutoff=cutoff,
            batch=batch,
            max_num_neighbors=max_num_neighbors,
        )

    return edge_index, edge_type


def get_neighbor_ids(data):
    """
    Takes the edge indices and returns dictionary mapping atom index to neighbor indices
    Note: this only includes atoms with degree > 1
    """
    batch_nbrs = deepcopy(data.neighbors)
    batch_nbrs = [obj[0] for obj in batch_nbrs]
    neighbors = batch_nbrs.pop(0)  # get first element
    n_atoms_per_mol = data.batch.bincount()  # get atom count per graph
    n_atoms_prev_mol = 0

    for i, n_dict in enumerate(batch_nbrs):
        new_dict = {}
        n_atoms_prev_mol += n_atoms_per_mol[i].item()
        for k, v in n_dict.items():
            new_dict[k + n_atoms_prev_mol] = v + n_atoms_prev_mol
        neighbors.update(new_dict)

    return neighbors


def signed_volume(local_coords):
    """
    Compute signed volume given ordered neighbor local coordinates
    From GeoMol

    :param local_coords: (n_tetrahedral_chiral_centers, 4, n_generated_confs, 3)
    :return: signed volume of each tetrahedral center
    (n_tetrahedral_chiral_centers, n_generated_confs)
    """
    v1 = local_coords[:, 0] - local_coords[:, 3]
    v2 = local_coords[:, 1] - local_coords[:, 3]
    v3 = local_coords[:, 2] - local_coords[:, 3]
    cp = v2.cross(v3, dim=-1)
    vol = torch.sum(v1 * cp, dim=-1)
    return torch.sign(vol)


def get_chiral_tensors(mol):
    """Only consider chiral atoms with 4 neighbors"""
    chiral_index = torch.tensor(
        [
            i
            for i, atom in enumerate(mol.GetAtoms())
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=torch.int32,
    ).view(
        1, -1
    )  # (1, n_chiral_centers)
    # (n_chiral_centers, 4)
    chiral_nbr_index = torch.tensor(
        [
            [n.GetIdx() for n in atom.GetNeighbors()]
            for atom in mol.GetAtoms()
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=torch.int32,
    ).view(
        1, -1
    )  # (1, n_chiral_centers * 4)
    # (n_chiral_centers,)
    chiral_tag = torch.tensor(
        [
            chirality[atom.GetChiralTag()]
            for atom in mol.GetAtoms()
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=torch.float32,
    )

    return chiral_index, chiral_nbr_index, chiral_tag
