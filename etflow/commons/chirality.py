from copy import deepcopy

import torch
from rdkit.Chem.rdchem import ChiralType

# similar to GeoMol
chirality = {
    ChiralType.CHI_TETRAHEDRAL_CW: -1.0,
    ChiralType.CHI_TETRAHEDRAL_CCW: 1.0,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0,
}


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
