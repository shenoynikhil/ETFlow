import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch_geometric.data import Batch, Data


@torch.no_grad()
def batched_sampling(
    model,
    data: Data,
    max_batch_size: int = 1,
    num_samples: int = 1,
    n_timesteps=50,
    seed: int = 42,
    device: str = "cpu",
    s_churn: float = 1.0,
    t_min: float = 1.0,
    t_max: float = 1.0,
    std: float = 1.0,
    sampler_type: str = "ode",
):
    if seed is not None:
        seed_everything(seed)

    sampled_pos = []

    for batch_start in range(0, num_samples, max_batch_size):
        # get batch_size
        batch_size = min(max_batch_size, num_samples - batch_start)
        # batch the data
        batched_data = Batch.from_data_list([data] * batch_size)

        # get one_hot, edge_index, batch
        (
            z,
            edge_index,
            batch,
            node_attr,
            chiral_index,
            chiral_nbr_index,
            chiral_tag,
        ) = (
            batched_data["atomic_numbers"].to(device),
            batched_data["edge_index"].to(device),
            batched_data["batch"].to(device),
            batched_data["node_attr"].to(device),
            batched_data["chiral_index"].to(device),
            batched_data["chiral_nbr_index"].to(device),
            batched_data["chiral_tag"].to(device),
        )

        with torch.no_grad():
            pos = model.sample(
                z=z,
                bond_index=edge_index,
                batch=batch,
                node_attr=node_attr,
                n_timesteps=n_timesteps,
                chiral_index=chiral_index,
                chiral_nbr_index=chiral_nbr_index,
                chiral_tag=chiral_tag,
                s_churn=s_churn,
                t_min=t_min,
                t_max=t_max,
                std=std,
                sampler_type=sampler_type,
            )

        # reshape to (num_samples, num_atoms, 3) using batch
        pos = pos.view(batch_size, -1, 3).cpu().detach().numpy()

        # append to generated_positions
        sampled_pos.append(pos)

    # concatenate generated_positions
    sampled_pos = np.concatenate(sampled_pos, axis=0)  # (num_samples, num_atoms, 3)

    return sampled_pos
