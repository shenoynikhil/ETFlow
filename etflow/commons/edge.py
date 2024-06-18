import numpy as np
import torch
from torch_cluster import radius_graph
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_sparse import coalesce

from .featurization import bond_to_feature_vector
from rdkit.Chem.rdchem import BondType as BT

BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}

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
