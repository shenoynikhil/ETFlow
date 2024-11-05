# allowable multiple choice node and edge features
from collections import defaultdict
from typing import Callable, Tuple

import datamol as dm
import torch
from datamol.types import Mol
from rdkit import Chem
from torch_geometric.data import Data

from .covmat import build_conformer
from .utils import atom_to_feature_vector, compute_edge_index, get_chiral_tensors


def get_mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return mol


def cache_decorator(func: Callable):
    """Decorator to handle caching logic."""

    def wrapper(self, smiles: str, *args, **kwargs):
        cache_key = func.__name__
        if smiles in self.cache and cache_key in self.cache[smiles]:
            return self.cache[smiles][cache_key]
        result = func(self, smiles, *args, **kwargs)
        self.cache[smiles][cache_key] = result
        return result

    return wrapper


class MoleculeFeaturizer:
    """A Featurizer Class for Molecules.
    - Give smiles, get mol objects, atom features, bond features, etc.
    - Smiles-based Caching to avoid recomputation.
    """

    def __init__(self):
        # smiles based cache
        self.cache = defaultdict(dict)

    def get_mol(self, smiles: str) -> Mol:
        return dm.to_mol(smiles, remove_hs=False, ordered=True)

    @cache_decorator
    def get_atom_features(self, smiles: str, use_ogb_feat: bool = True) -> torch.Tensor:
        # compute atom features
        mol = self.get_mol(smiles)
        atom_features = self.get_atom_features_from_mol(mol, use_ogb_feat=use_ogb_feat)
        return atom_features

    @cache_decorator
    def get_atomic_numbers(self, smiles: str) -> torch.Tensor:
        # compute atomic numbers
        mol = self.get_mol(smiles)
        atomic_numbers = self.get_atomic_numbers_from_mol(mol)
        return atomic_numbers

    def get_atomic_numbers_from_mol(self, mol: Mol) -> torch.Tensor:
        atomic_numbers = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=torch.int32,
        )
        return atomic_numbers

    def get_atom_features_from_mol(
        self, mol: Mol, use_ogb_feat: bool = True
    ) -> torch.Tensor:
        if use_ogb_feat:
            atom_features = torch.tensor(
                [atom_to_feature_vector(atom) for atom in mol.GetAtoms()],
                dtype=torch.float32,
            )
        else:
            atom_features = torch.tensor(
                [atom.GetFormalCharge() for atom in mol.GetAtoms()],
                dtype=torch.float32,
            ).view(-1, 1)
        return atom_features

    @cache_decorator
    def get_chiral_centers(self, smiles: str) -> torch.Tensor:
        # compute chiral centers
        mol = self.get_mol(smiles)
        chiral_index, chiral_nbr_index, chiral_tag = self.get_chiral_centers_from_mol(
            mol
        )

        self.cache[smiles]["chiral_centers"] = (
            chiral_index,
            chiral_nbr_index,
            chiral_tag,
        )
        return chiral_index, chiral_nbr_index, chiral_tag

    def get_chiral_centers_from_mol(self, mol: Mol) -> torch.Tensor:
        chiral_index, chiral_nbr_index, chiral_tag = get_chiral_tensors(mol)
        return chiral_index, chiral_nbr_index, chiral_tag

    @cache_decorator
    def get_mol_with_conformer(self, smiles: str, positions: torch.Tensor) -> Mol:
        mol = self.get_mol(smiles)
        mol.AddConformer(build_conformer(positions))
        return mol

    @cache_decorator
    def get_edge_index(
        self, smiles: str, use_edge_feat: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns edge index and edge attributes for a given smiles."""
        # compute edge index
        mol = self.get_mol(smiles)
        edge_index, edge_attr = self.get_edge_index_from_mol(
            mol, use_edge_feat=use_edge_feat
        )

        self.cache[smiles]["edge_index"] = edge_index
        self.cache[smiles]["edge_attr"] = edge_attr
        return edge_index, edge_attr

    def get_edge_index_from_mol(
        self, mol: Mol, use_edge_feat: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns edge index and edge attributes for a given mol object."""
        edge_index, edge_attr = compute_edge_index(mol, with_edge_attr=use_edge_feat)
        return edge_index, edge_attr

    def get_data_from_smiles(self, smiles: str) -> Data:
        mol = get_mol_from_smiles(smiles)  # added hs
        smiles_changed = dm.to_smiles(
            mol,
            canonical=False,
            explicit_hs=True,
            with_atom_indices=True,
            isomeric=True,
        )
        node_attr = self.get_atom_features_from_mol(mol, True)
        chiral_index, chiral_nbr_index, chiral_tag = self.get_chiral_centers_from_mol(
            mol
        )
        edge_index, edge_attr = self.get_edge_index_from_mol(mol, False)
        atomic_numbers = self.get_atomic_numbers_from_mol(mol)

        graph = Data(
            atomic_numbers=atomic_numbers,
            smiles=smiles_changed,
            edge_index=edge_index,
            chiral_index=chiral_index,
            chiral_nbr_index=chiral_nbr_index,
            chiral_tag=chiral_tag,
            node_attr=node_attr,
            edge_attr=edge_attr,
        )
        return graph
