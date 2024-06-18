from typing import Tuple

import numpy as np
import torch
from torch.nn.functional import pad
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, get_laplacian, to_dense_adj
from torch_scatter import scatter

from etflow.commons import extend_graph_order_radius


def center_pos(pos, batch):
    pos_center = pos - scatter(pos, batch, dim=0, reduce="mean")[batch]
    return pos_center


def linear_schedule(low, high, max_steps, total_steps) -> torch.Tensor:
    schedule = torch.linspace(low, high, steps=max_steps)

    if max_steps < total_steps:
        pad_size = abs(total_steps - max_steps)
        schedule = pad(schedule, pad=(0, pad_size), mode="constant", value=high)

    return schedule


def center_of_mass(x, dim=0, batch=None):
    num_nodes = x.size(0)

    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

    x_com = scatter(x, batch, dim=dim, reduce="mean")[batch]
    return x - x_com


def assert_zero_mean(x: torch.Tensor, batch: torch.Tensor, eps=1e-10) -> bool:
    largest_value = x.abs().max().item()
    a = scatter(x, batch, dim=0, reduce="mean") if batch is not None else x.mean(dim=0)
    error = a.abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"


def extend_bond_index(
    pos: torch.Tensor,
    bond_index: torch.Tensor,
    batch: torch.Tensor,
    bond_attr: torch.Tensor,
    device: torch.device,
    one_hot: bool = False,
    one_hot_types: int = 5,
    cutoff: float = 10.0,
    max_num_neighbors: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if bond_attr is None:
        bond_type = torch.ones(bond_index.shape[1], dtype=torch.long, device=device)
        # all molecular graph edges are type 1, radius based become 0
    else:
        bond_type = bond_attr.view(-1).long() + 1  # we reserve 0 for radius based edges
        assert (
            bond_type.shape[0] == bond_index.shape[1]
        ), "Edge type should have same shape as number of edges."

    edge_index, edge_type = extend_graph_order_radius(
        pos=pos,
        edge_index=bond_index,
        edge_type=bond_type,
        batch=batch,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
        extend_radius=True,
        extend_order=False,
    )
    assert (
        bond_index.shape[1] == (edge_type > 0).sum().item()
    ), "Edge Type should be greater than 0 when edge is a molecular bond."

    # make one_hot if provided
    if one_hot:
        # +1 to account for radius based edges
        edge_type = torch.nn.functional.one_hot(
            edge_type, num_classes=one_hot_types + 1
        ).float()

    return edge_index, edge_type


# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def polynomial_schedule(timesteps: int, s: float = 1e-4, power: int = 2) -> np.ndarray:
    """Polynomial Schedule
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    alpha_t = 1, when t = 0 and alpha_t = 0, when t = T.
    """
    steps = timesteps + 1
    clip_value = 0.001
    # sample T + 1 evenly spaced points from [0, T] (both inclusive)
    x = np.linspace(0, steps, steps)
    # compute polynomial at each point, shape: (T + 1, )
    alphas2 = (1 - np.power(x / steps, power)) ** 2
    # add 1 to the beginning of the array, shape: (T + 2, 1)
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    # Compute alpha_{t|s} = alpha_t / alpha_s where s + 1 = t, shape: (T + 1, )
    alphas_step = alphas2[1:] / alphas2[:-1]
    # Clip values between 0.001 and 1.0
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    # Compute cumulative product of alpha_{t|s} for each t, shape: (T, )
    alphas2 = np.cumprod(alphas_step, axis=0)

    # Not sure why this is done, TODO: check
    precision = 1 - 2 * s
    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(
    timesteps: int, s: float = 0.008, raise_to_power: float = 1
) -> np.ndarray:
    """Cosine Schedule
    Proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for
    predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule: str, timesteps: int, precision: float):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(timesteps)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


def gaussian_KL_for_dimension(
    q_mu: torch.Tensor,
    q_sigma: torch.Tensor,
    p_mu: torch.Tensor,
    p_sigma: torch.Tensor,
    d: torch.Tensor,
    batch: torch.Tensor,
):
    """Computes the KL distance between two normal distributions.

    Args:
        q_mu: Mean of distribution q. (N, 3)
        q_sigma: Standard deviation of distribution q. (B, 1)
        p_mu: Mean of distribution p. (N, 3)
        p_sigma: Standard deviation of distribution p. (B, 1)
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    mu_norm2 = scatter(
        torch.sum((q_mu - p_mu) ** 2, dim=1), batch, reduce="sum"
    )  # (B, 1)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return (
        d * torch.log(p_sigma / q_sigma)
        + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2)
        - 0.5 * d
    )  # (B, 1)


def unsqueeze_like(x: torch.Tensor, target: torch.Tensor):
    shape = (x.size(0), *([1] * (target.dim() - 1)))
    return x.view(shape)


"""
Following code adapted from HarmonicFlow
https://github.com/HannesStark/FlowSite/blob/main/utils/diffusion.py
"""


class HarmonicPotential(MessagePassing):
    def __init__(self, alpha=1, normalize=False):
        super(HarmonicPotential, self).__init__()
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0)).to(x.device).long()

        # multiple by 0.5 since undirected edges
        energy = 0.5 * self.propagate(edge_index, x=x)

        # normalize
        if self.normalize:
            # divide by 2 since edge index is undirected and we
            # take account double counting
            node_degree = 0.5 * degree(edge_index[0], num_nodes=x.size(0))
            energy = (1 / node_degree) * energy

        energy = 0.5 * self.alpha * scatter(energy, index=batch)
        return energy.view(-1, 1)

    def message(self, x_i, x_j):
        return torch.norm(x_i - x_j, p=2, dim=-1) ** 2

    def aggregate(
        self,
        x,
        index,
    ):
        return scatter(x, index, dim=-1)

    def update(self, inputs):
        return inputs


class HarmonicSampler:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.eig_val_cache = {}
        self.eig_vec_cache = {}

    def diagonalize(self, n_nodes, edges=[], batch=None, smiles=None):
        a = self.alpha * torch.ones((edges.shape[0],), device=edges.device)
        edge_index, edge_weight = get_laplacian(
            edges.T,
            a,
            num_nodes=n_nodes,
        )

        H = to_dense_adj(
            edge_index=edge_index, edge_attr=edge_weight, max_num_nodes=n_nodes
        ).squeeze()

        if batch is None:
            D, P = torch.linalg.eigh(H)
            return D, P

        Ds, Ps = [], []
        batch_size = batch.max() + 1

        for i in range(batch_size):
            idx = torch.where(batch == i)[0]
            start = idx.min()
            end = idx.max() + 1

            D, P = None, None
            if smiles is not None:
                D, P = self.check_cache(smiles[i])

                if (D is not None) and (P is not None):
                    D = D.to(edge_index.device)
                    P = P.to(edge_index.device)

            if (D is None) or (P is None):
                D, P = torch.linalg.eigh(H[start:end, start:end])

                if smiles is not None:
                    self.eig_val_cache[smiles[i]] = D.cpu()
                    self.eig_vec_cache[smiles[i]] = P.cpu()

            Ds.append(D)
            Ps.append(P)

        return torch.cat(Ds), torch.block_diag(*Ps)

    def check_cache(self, smiles):
        D = self.eig_val_cache.get(smiles, None)
        P = self.eig_vec_cache.get(smiles, None)
        return D, P

    def sample(self, size, edge_index, batch=None, smiles=None):
        # transpose if (2, n_edges)
        if edge_index.size(0) == 2:
            edge_index = edge_index.T

        n_nodes = size[0]
        D, P = self.diagonalize(
            n_nodes=n_nodes, edges=edge_index, batch=batch, smiles=smiles
        )

        # get starting index per sample in batch
        start_index = 0
        if batch is not None:
            _, counts = torch.unique(batch, return_counts=True)
            cum_sum = counts.cumsum(0)[:-1]
            zero = torch.zeros(1).to(D.device)
            start_index = torch.concat((zero, cum_sum)).long()

        std = 1.0 / torch.sqrt(D)
        std[start_index] = 0.0

        noise = torch.randn(size).to(D.device)
        noise = std[:, None] * noise
        noise[noise.isnan()] = 0.0
        sample = P @ (noise)

        return sample

    def energy(self, x, edge_index, batch=None, smiles=None):
        n_nodes = x.size(0)
        x = center_of_mass(x)

        if batch is None:
            batch = torch.zeros(n_nodes).to(x.device).long()

        if edge_index.size(0) == 2:
            edge_index = edge_index.T

        D, P = self.diagonalize(n_nodes, edges=edge_index, batch=batch, smiles=smiles)

        start_index = 0
        if batch is not None:
            _, counts = torch.unique(batch, return_counts=True)
            cum_sum = counts.cumsum(0)[:-1]
            zero = torch.zeros(1).to(D.device)
            start_index = torch.concat((zero, cum_sum)).long()

        energy_unpooled = D[:, None] * (P.T @ x) ** 2
        energy_unpooled[start_index] = 0.0
        energy_unpooled = energy_unpooled.sum(-1)
        energy = 0.5 * scatter(energy_unpooled, batch)

        return energy.view(-1, 1)


def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor(
            [[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float
        )
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def rmsd_align(pos, ref_pos, batch):
    aligned_pos = []
    batch_size = batch.max() + 1
    for i in range(batch_size):
        index = torch.where(batch == i)[0]
        pos_i = pos[index]
        ref_pos_i = ref_pos[index]
        R, t = find_rigid_alignment(pos_i, ref_pos_i)

        pos_i = (R @ pos_i.T).T + t
        aligned_pos.append(pos_i)

    return torch.concat(aligned_pos, dim=0)
