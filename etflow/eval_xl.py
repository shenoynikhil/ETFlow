"""Script to generate and evaluate samples on GEOM-XL from a trained diffusion model.

Usage
```bash
# requires config for loading model
# and checkpoint for loading model weights
python etflow/eval_xl.py \
    --config /path/to/config.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --batch_size 16 \
    --nsteps 50
```
"""
import argparse
import datetime
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from lightning import seed_everything
from loguru import logger as log
from torch_geometric.data import Batch, Data
from tqdm import tqdm

import wandb
from etflow.commons import (
    atom_to_feature_vector,
    compute_edge_index,
    get_atomic_number_and_charge,
    get_chiral_tensors,
    get_lpe,
    load_pkl,
    save_pkl,
)
from etflow.utils import instantiate_model, read_yaml

torch.set_float32_matmul_precision("high")


def get_data(mol, use_lpe=False, lpe_k=4):
    """Convert mol object to Data object"""
    x = get_atomic_number_and_charge(mol)
    atomic_numbers = torch.from_numpy(x[:, 0]).int()
    edge_index, _ = compute_edge_index(mol)
    edge_index = edge_index.long()
    node_attr = torch.tensor(
        [atom_to_feature_vector(atom) for atom in mol.GetAtoms()],
        dtype=torch.float32,
    )
    chiral_index, chiral_nbr_index, chiral_tag = get_chiral_tensors(mol)
    if use_lpe:
        lpe = get_lpe(
            Data(edge_index=edge_index),
            num_nodes=x.shape[0],
            k=lpe_k,
        )
        node_attr = torch.cat([node_attr, lpe], dim=-1)
    return Data(
        atomic_numbers=atomic_numbers,
        edge_index=edge_index,
        node_attr=node_attr,
        chiral_index=chiral_index,
        chiral_nbr_index=chiral_nbr_index,
        chiral_tag=chiral_tag,
    )


BASE_PATH = "/nfs/scratch/students/data/geom/preprocessed/"


def get_datatime():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def cuda_available():
    return torch.cuda.is_available()


def main(
    config: str,
    checkpoint_path: str,
    nsteps: int,
    batch_size: int,
    eps: float,
    debug: bool,
    unfiltered: bool,
):
    seed = config.get("seed", 42)
    seed_everything(seed)

    if cuda_available():
        log.info("CUDA is available. Using GPU for sampling.")
        device = torch.device("cuda")
    else:
        log.warning("CUDA is not available. Using CPU for sampling.")
        device = torch.device("cpu")

    # load mols
    smiles_df = pd.read_csv(
        osp.join(
            BASE_PATH,
            "XL/test_smiles.csv" if unfiltered else "XL/test_smiles_filtered.csv",
        )
    )
    smiles_list = smiles_df["corrected_smiles"].tolist()
    counts = smiles_df["n_conformers"].tolist()
    mols = load_pkl(osp.join(BASE_PATH, "XL/test_mols.pkl"))

    # instantiate datamodule and model
    model_type = config["model"]
    model = instantiate_model(config["model"], config["model_args"])

    # load model weights
    log.info(f"Loading model weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)

    # check if we need to use lpe
    lpe_k = None
    use_lpe = False
    if (
        "use_lpe" in config["datamodule_args"]["dataset_args"]
        and config["datamodule_args"]["dataset_args"]["use_lpe"]
    ):
        use_lpe = True
        lpe_k = config["datamodule_args"]["dataset_args"].get("lpe_k", 4)
        log.info(f"Using LPE with k={lpe_k}")

    # move to device
    model = model.to(device)
    model = model.eval()

    # max batch size
    max_batch_size = batch_size

    # load indices
    for idx, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list)):
        # get data for batch_size
        log.info(f"Generating conformers for molecule: {smiles}")

        # calculate number of samples to generated
        count = counts[idx]
        num_samples = 2 * count

        # get molecular graph
        mol_obj = mols[smiles][0]
        data = get_data(mol_obj, use_lpe=use_lpe, lpe_k=lpe_k)

        # we would want (num_samples, num_nodes, 3)
        generated_positions = []

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

            chiral_index = batched_data["chiral_index"].to(device)
            chiral_nbr_index = batched_data["chiral_nbr_index"].to(device)
            chiral_tag = batched_data["chiral_tag"].to(device)

            if model_type == "BaseSFM":
                # generate samples
                pos = model.sample(
                    z,
                    edge_index,
                    batch,
                    node_attr=node_attr,
                    n_timesteps=nsteps,
                    eps=eps,
                    use_sde=True,
                )
            else:
                # generate samples
                pos = model.sample(
                    z,
                    edge_index,
                    batch,
                    node_attr=node_attr,
                    n_timesteps=nsteps,
                    chiral_index=chiral_index,
                    chiral_nbr_index=chiral_nbr_index,
                    chiral_tag=chiral_tag,
                )

            # reshape to (num_samples, num_atoms, 3) using batch
            pos = pos.view(batch_size, -1, 3).cpu().detach().numpy()

            # append to generated_positions
            generated_positions.append(pos)

        # concatenate generated_positions
        generated_positions = np.concatenate(
            generated_positions, axis=0
        )  # (num_samples, num_atoms, 3)

        # save to file
        if not debug:
            path = osp.join(output_dir, f"{idx}.pkl")
            log.info(
                f"Saving generated positions to file for smiles {smiles} at {path}"
            )
            save_pkl(path, generated_positions)

    # compile all generate pkl into a single file
    log.info("Compile all generated pickle files into a single file")
    l = set([int(x.split(".pkl")[0]) for x in os.listdir(output_dir)])
    log.info(f"Number of generated files: {len(l)}")

    # create pos_gen
    data_list = {}
    for idx, smiles in tqdm(enumerate(smiles_list)):
        gt_mol_list = mols[smiles]
        for gt_mol in gt_mol_list:
            pos_item = gt_mol.GetConformer(0).GetPositions()
            if smiles not in data_list:
                data_list[smiles] = Data(smiles=smiles, pos_ref=pos_item, rdmol=gt_mol)
            else:
                # just update pos_ref
                pos_ref_collected = data_list[smiles].pos_ref
                data_list[smiles].pos_ref = np.concatenate(
                    [pos_ref_collected, pos_item]
                )

        # grab generated positions from file
        pos_gen = load_pkl(f"{output_dir}/{idx}.pkl")
        data_list[smiles].pos_gen = pos_gen

    save_pkl(os.path.join(output_dir, "generated_files.pkl"), list(data_list.values()))


if __name__ == "__main__":
    # argparse checkpoint path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint", "-k", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=False, default="logs/")
    parser.add_argument("--batch_size", "-b", type=int, required=False, default=16)
    parser.add_argument("--nsteps", "-n", type=int, required=False, default=50)
    parser.add_argument("--unfiltered", "-u", action="store_true")
    parser.add_argument(
        "--eps",
        "-e",
        type=float,
        default=1.0,
        required=False,
        help="Noise coefficient for SFM",
    )
    parser.add_argument("--debug", "-d", action="store_true")

    args = parser.parse_args()

    # debug mode
    debug = args.debug
    log.info(f"Debug mode: {debug}")

    # read config
    config_path = args.config
    assert osp.exists(config_path), "Config path does not exist."
    log.info(f"Loading config from: {config_path}")
    config = read_yaml(config_path)
    task_name = config.get("task_name", "default")

    # start wandb
    if not debug:
        log.info("Starting wandb...")
        wandb.init(
            project="Energy-Aware-MCG",
            entity="doms-lab",
            name=f"(GEOM-XL) Sample Generation: {task_name}",
        )

        # log experiment info
        log_dict = {
            "config": config_path,
            "checkpoint": args.checkpoint,
            "debug": debug,
            "unfiltered": args.unfiltered,
            "nsteps": args.nsteps,
        }

        wandb.log(log_dict)

    # get checkpoint path
    checkpoint_path = args.checkpoint
    assert osp.exists(checkpoint_path), "Checkpoint path does not exist."

    # setup output directory for storing samples
    output_dir = osp.join(
        args.output_dir,
        f"samples/{task_name}/{get_datatime()}/flow_nsteps_{args.nsteps}_geom_xl",
    )
    if not debug:
        os.makedirs(output_dir, exist_ok=True)

    main(
        config,
        checkpoint_path,
        nsteps=args.nsteps,
        batch_size=args.batch_size,
        eps=args.eps,
        debug=debug,
        unfiltered=args.unfiltered,
    )
