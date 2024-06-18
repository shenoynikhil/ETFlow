# allowable multiple choice node and edge features
from typing import Tuple

import datamol as dm
import numpy as np
import torch
from datamol.types import Mol
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType

from .chirality import get_chiral_tensors
from .covmat import build_conformer
from .edge import compute_edge_index

# similar to GeoMol
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
