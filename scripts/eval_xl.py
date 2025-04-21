"""Script to generate and evaluate samples on GEOM-XL from a trained diffusion model.

Usage
```bash
python etflow/eval_xl.py \
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

import numpy as np
import torch
from loguru import logger as log
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from utils import instantiate_model, read_yaml

import wandb
from etflow.commons import MoleculeFeaturizer, get_base_data_dir, load_pkl, save_pkl

DATA_DIR = get_base_data_dir()
torch.set_float32_matmul_precision("high")
mol_feat = MoleculeFeaturizer()


def get_data(mol):
    """Convert mol object to Data object"""
    atomic_numbers = mol_feat.get_atomic_numbers_from_mol(mol)
    edge_index, _ = mol_feat.get_edge_index_from_mol(mol)
    node_attr = mol_feat.get_atom_features_from_mol(mol)
    chiral_index, chiral_nbr_index, chiral_tag = mol_feat.get_chiral_centers_from_mol(
        mol
    )

    return Data(
        atomic_numbers=atomic_numbers,
        edge_index=edge_index,
        node_attr=node_attr,
        chiral_index=chiral_index,
        chiral_nbr_index=chiral_nbr_index,
        chiral_tag=chiral_tag,
    )


def get_datatime():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main(config: dict, checkpoint_path: str, output_dir: str, debug: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using {device} for sampling.")

    # load mols
    mols_path = osp.join(DATA_DIR, "XL/test_mols.pkl")
    mols = load_pkl(mols_path)

    # instantiate model
    model = instantiate_model(config["model"], config["model_args"])

    # load model weights
    log.info(f"Loading model weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)

    # move to device
    model = model.to(device)
    model.eval()

    # max batch size
    max_batch_size = config["eval_args"]["batch_size"]

    # load indices
    data_list = {}

    smiles_list = list(mols.keys())
    for _, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list)):
        # get data for batch_size
        log.info(f"Generating conformers for molecule: {smiles}")

        # get molecular graph
        mol_obj = mols[smiles][0]
        count = len(mols[smiles])
        num_samples = 2 * count
        data = get_data(mol_obj)

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

            # reshape to (num_samples, num_atoms, 3) using batch
            pos = pos.view(batch_size, -1, 3).cpu().detach().numpy()
            pos_gen.append(pos)

        # concatenate generated_positions
        pos_gen = np.concatenate(pos_gen, axis=0)

        # Get reference positions
        gt_mol_list = mols[smiles]
        pos_ref = []
        for gt_mol in gt_mol_list:
            pos_ref.append(gt_mol.GetConformer(0).GetPositions())
        pos_ref = np.stack(pos_ref, axis=0)

        data_list[smiles] = Data(
            smiles=smiles, pos_ref=pos_ref, rdmol=mol_obj, pos_gen=pos_gen
        )
        if debug:
            break

    if not debug and data_list:
        save_pkl(
            os.path.join(output_dir, "generated_files.pkl"), list(data_list.values())
        )


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
            name=f"(GEOM-XL) Sample Generation: {task_name}",
        )

        # log experiment info
        wandb.log(
            {
                "config": args.config,
                "checkpoint": args.checkpoint,
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
