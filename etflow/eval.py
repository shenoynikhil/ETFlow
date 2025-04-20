"""Script to generate samples from a trained model.

Usage
```bash
python etflow/eval.py \
    --config=</path/to/config> \
    --checkpoint=</path/to/checkpoint> \
    --output_dir=</path/to/output_dir> \
    [--debug]
```
"""

import argparse
import datetime
import os
import os.path as osp
import time

import numpy as np
import torch

# from lightning import seed_everything
from loguru import logger as log
from torch_geometric.data import Batch, Data
from tqdm import tqdm

import wandb
from etflow.commons import save_pkl
from etflow.data import EuclideanDataset
from etflow.models import BaseFlow
from etflow.utils import instantiate_model, read_yaml

torch.set_float32_matmul_precision("high")


def get_datatime():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main(config: dict, checkpoint_path: str, output_dir: str, debug: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using {device} for sampling.")

    # instantiate datamodule and model
    dataset = EuclideanDataset(
        data_dir=config["datamodule_args"]["data_dir"],
        partition=config["datamodule_args"]["partition"],
        split="test",
    )
    model = instantiate_model(config["model"], config["model_args"])

    # load model weights
    log.info(f"Loading model weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)

    # move to device
    model: BaseFlow = model.to(device)
    model.eval()

    # max batch size
    max_batch_size = config["eval_args"]["batch_size"]

    # load indices
    num_test_samples = len(dataset)
    data_list = {}
    times = []

    for idx in tqdm(range(num_test_samples)):
        data = dataset[idx]

        # get data for batch_size
        smiles = data.smiles
        log.info(f"Generating conformers for molecule: {smiles}")

        # calculate number of samples to generated
        pos_ref: torch.Tensor = torch.load(dataset.data_files[idx]).pos.cpu().numpy()
        count = pos_ref.shape[0]  # number of conformers
        num_samples = 2 * count
        pos_gen = []

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

            # get time-estimate
            start = time.time()
            with torch.no_grad():
                # generate samples
                pos = model.sample(
                    z,
                    edge_index,
                    batch,
                    node_attr=node_attr,
                    chiral_index=chiral_index,
                    chiral_nbr_index=chiral_nbr_index,
                    chiral_tag=chiral_tag,
                    **config["eval_args"]["sampler_args"],
                )
            end = time.time()
            times.append((end - start) / batch_size)  # store time per conformer

            # reshape to (num_samples, num_atoms, 3) using batch
            pos = pos.view(batch_size, -1, 3).cpu().detach().numpy()

            # append to generated_positions
            pos_gen.append(pos)

        # concatenate generated_positions: (num_samples, num_atoms, 3)
        pos_gen = np.concatenate(pos_gen, axis=0)

        data_list[smiles] = Data(
            smiles=smiles, pos_ref=pos_ref, rdmol=data.mol, pos_gen=pos_gen
        )
        if debug:
            break

    if not debug:
        save_pkl(
            os.path.join(output_dir, "generated_files.pkl"), list(data_list.values())
        )

        # log time per conformer
        wandb.log({"time_per_conformer": np.mean(times)})
        save_pkl(os.path.join(output_dir, "times.pkl"), times)


if __name__ == "__main__":
    # argparse checkpoint path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint", "-k", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=False, default="logs/")
    parser.add_argument("--debug", "-d", action="store_true")

    args = parser.parse_args()

    # debug mode
    debug = args.debug
    log.info(f"Debug mode: {debug}")

    # read config
    assert osp.exists(args.config), "Config path does not exist."
    log.info(f"Loading config from: {args.config}")
    config = read_yaml(args.config)
    task_name = config.get("task_name", "default")

    # start wandb
    if not debug:
        log.info("Starting wandb...")
        wandb.init(
            project="Energy-Aware-MCG",
            entity="doms-lab",
            name=f"Sample Generation: {task_name}",
        )

        # log experiment info
        wandb.log(
            {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "dataset_type": config["datamodule_args"]["partition"],
                "debug": debug,
            }
        )

    # get checkpoint path
    checkpoint_path = args.checkpoint
    assert osp.exists(checkpoint_path), "Checkpoint path does not exist."

    # setup output directory for storing samples
    output_dir = osp.join(
        args.output_dir,
        f"samples/{task_name}/{get_datatime()}",
    )
    if not debug:
        os.makedirs(output_dir, exist_ok=True)

    main(config, checkpoint_path, output_dir, debug=debug)
