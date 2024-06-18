import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch_geometric.data import Batch


@torch.no_grad()
def batched_sampling(
    model, data, max_batch_size, num_samples, n_timesteps=50, seed=None, device="cpu"
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
        z, edge_index, batch, node_attr = (
            batched_data["atomic_numbers"].to(device),
            batched_data["edge_index"].to(device),
            batched_data["batch"].to(device),
            batched_data["node_attr"].to(device),
        )

        with torch.no_grad():
            pos = model.sample(
                z=z,
                bond_index=edge_index,
                batch=batch,
                node_attr=node_attr,
                n_timesteps=n_timesteps,
            )

        # reshape to (num_samples, num_atoms, 3) using batch
        pos = pos.view(batch_size, -1, 3).cpu().detach().numpy()

        # append to generated_positions
        sampled_pos.append(pos)

    # concatenate generated_positions
    sampled_pos = np.concatenate(sampled_pos, axis=0)  # (num_samples, num_atoms, 3)

    return sampled_pos
