import argparse
import datetime
import os
import os.path as osp
from copy import deepcopy

import numpy as np
import torch
import wandb
from lightning.pytorch import seed_everything
from loguru import logger as log
from tqdm import tqdm

from etflow.commons import (
    batched_sampling,
    build_conformer,
    save_pkl,
    xtb_energy,
    xtb_optimize,
)
from etflow.utils import instantiate_dataset, instantiate_model, read_yaml

torch.set_float32_matmul_precision("high")


def get_datatime():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def cuda_available():
    return torch.cuda.is_available()


def prop_mean(prop, energy):
    kT = 0.592

    weights = -energy / kT
    weights = np.exp(weights - weights.max())
    weights = weights / weights.sum()
    return (weights * prop).sum()


def compute_props(mol_list, xtb_path, e0=None, opt=False):
    energy_results = []
    dipole_results = []
    gap_results = []

    for i in range(len(mol_list)):
        mol = mol_list[i]
        if opt:
            xtb_optimize(mol, "normal", path_xtb=xtb_path)

        res = xtb_energy(mol=mol, path_xtb=xtb_path, dipole=True)
        energy, dipole, gap = (
            res["energy"],
            res["dipole"],
            res["gap"],
        )

        if e0 is not None:
            energy -= e0

        energy_results.append(energy)
        dipole_results.append(dipole)
        gap_results.append(gap)

    energy_results = np.asarray(energy_results)
    dipole_results = np.asarray(dipole_results)
    gap_results = np.asarray(gap_results)

    return {
        "energy": energy_results,
        "dipole": dipole_results,
        "gap": gap_results,
    }


def build_mol(mol, pos):
    mol_copy = deepcopy(mol)
    mol_copy.RemoveAllConformers()
    mol_copy.AddConformer(build_conformer(pos))
    return mol_copy


def compute_metrics(gen_props, gt_props):
    num_rand_samples = len(gen_props)

    gt_energy_list = []
    gen_energy_list = []

    gt_dipole_list = []
    gen_dipole_list = []

    gt_gap_list = []
    gen_gap_list = []

    gt_energy_min_list = []
    gen_energy_min_list = []

    for i in range(num_rand_samples):
        gt_prop = gt_props[i]
        gen_prop = gen_props[i]

        gt_energy = gt_prop["energy"]
        gen_energy = gen_prop["energy"]

        gt_energy_list.append(prop_mean(gt_energy, gt_energy))
        gen_energy_list.append(prop_mean(gen_energy, gen_energy))

        gt_dipole = gt_prop["dipole"]
        gen_dipole = gen_prop["dipole"]

        gt_dipole_list.append(prop_mean(gt_dipole, gt_energy))
        gen_dipole_list.append(prop_mean(gen_dipole, gen_energy))

        gt_gap = gt_prop["gap"]
        gen_gap = gen_prop["gap"]

        gt_gap_list.append(prop_mean(gt_gap, gt_energy))
        gen_gap_list.append(prop_mean(gen_gap, gen_energy))

        gt_energy_min_list.append(np.min(gt_energy))
        gen_energy_min_list.append(np.min(gen_energy))

    gt_energy = np.asarray(gt_energy_list)
    gen_energy = np.asarray(gen_energy_list)

    gt_dipole = np.asarray(gt_dipole_list)
    gen_dipole = np.asarray(gen_dipole_list)

    gt_gap = np.asarray(gt_gap_list)
    gen_gap = np.asarray(gen_gap_list)

    gt_energy_min = np.asarray(gt_energy_min_list)
    gen_energy_min = np.asarray(gen_energy_min_list)

    energy_med_abs_error = np.median(abs(gt_energy - gen_energy))
    dipole_med_abs_error = np.median(abs(gt_dipole - gen_dipole))
    gap_med_abs_error = np.median(abs(gt_gap - gen_gap))
    min_energy_med_abs_error = np.median(abs(gt_energy_min - gen_energy_min))

    print(f"Energy Median Abs Error: {energy_med_abs_error}")
    print(f"Dipole Median Abs Error: {dipole_med_abs_error}")
    print(f"Gap Median Abs Error: {gap_med_abs_error}")
    print(f"Min Energy Median Abs Error: {min_energy_med_abs_error}")

    props_metrics = {
        "Energy Median Error": energy_med_abs_error,
        "Dipole Median Error": dipole_med_abs_error,
        "Gap Median Error": gap_med_abs_error,
        "Min Energy Median Error": min_energy_med_abs_error,
    }

    return props_metrics


def main(
    config: dict,
    checkpoint_path: str,
    xtb_path: str,
    indices: np.ndarray,
    counts: np.ndarray,
    nsteps: int,
    max_batch_size: int = 512,
    num_workers: int = 1,
    debug: bool = False,
    seed: int = 42,
):
    # seed everything for reproducibility
    log.info(f"setting seed to {seed}")
    seed_everything(seed)

    if cuda_available():
        log.info("CUDA is available. Using GPU for sampling.")
        device = torch.device("cuda")
    else:
        log.warning("CUDA is not available. Using CPU for sampling.")
        device = torch.device("cpu")

    # instantiate datamodule and model
    dataset = instantiate_dataset(
        config["datamodule_args"]["dataset"], config["datamodule_args"]["dataset_args"]
    )

    model = instantiate_model(config["model"], config["model_args"])

    # load model weights
    log.info(f"Loading model weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)

    # move to device
    model = model.to(device)
    model = model.eval()

    random_index = np.random.randint(low=0, high=len(indices), size=(100,))

    # load indices
    gt_props_list = []
    gen_props_list = []

    for i, rand_idx in tqdm(enumerate(random_index), total=len(random_index)):
        idx = indices[rand_idx]
        data = dataset[idx]

        # get data for batch_size
        smiles = data.smiles
        log.info(f"Generating conformers for molecule: {smiles}")

        count = counts[rand_idx]

        # calculate number of samples to generated
        num_samples = min(2 * count, 32)
        num_gt_samples = min(count, 32)

        # we would want (num_samples, num_nodes, 3)
        sampled_positions = batched_sampling(
            model=model,
            data=data,
            max_batch_size=max_batch_size,
            num_samples=num_samples,
            n_timesteps=nsteps,
            device=device,
        )

        log.info(f"Optimizing conformers and compute properties for molecule: {smiles}")

        properties = {}

        mol = data.mol
        gt_mol_list = [
            build_mol(mol, pos=dataset[idx + j].pos) for j in range(num_gt_samples)
        ]
        gen_mol_list = [
            build_mol(mol, pos=sampled_positions[j]) for j in range(num_samples)
        ]

        gt_props = compute_props(mol_list=gt_mol_list, xtb_path=xtb_path)
        gen_props = compute_props(mol_list=gen_mol_list, xtb_path=xtb_path, opt=True)

        properties["gt_props"] = gt_props
        properties["gen_props"] = gen_props
        properties["index"] = idx
        properties["smiles"] = data.smiles

        gt_props_list.append(gt_props)
        gen_props_list.append(gen_props)

        if debug:
            break

        path = osp.join(output_dir, f"{idx}.pkl")
        log.info(f"Saving generated positions to file for smiles {smiles} at {path}")

        save_pkl(path, properties)

    props_metrics = compute_metrics(gen_props=gen_props_list, gt_props=gt_props_list)
    wandb.run.log(props_metrics)


if __name__ == "__main__":
    # argparse checkpoint path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint", "-k", type=str, required=True)
    parser.add_argument("--xtb_path", "-x", type=str, required=True)
    parser.add_argument("--count_indices", "-i", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=False, default="logs/")
    parser.add_argument("--nsteps", "-n", type=int, required=False, default=100)
    parser.add_argument("--num_workers", "-w", type=int, required=False, default=1)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=42, help="seed")
    args = parser.parse_args()

    # debug mode
    debug = args.debug
    log.info(f"Debug mode: {debug}")

    num_workers = args.num_workers

    xtb_path = args.xtb_path
    assert osp.exists(xtb_path), "xTB path does not exist."

    # get nsteps
    nsteps = args.nsteps
    log.info(f"Number of steps (if using Flow): {nsteps}")

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
            name=f"Ensemble Properties Evaluation: {task_name}",
        )

        # log experiment info
        log_dict = {
            "config": config_path,
            "checkpoint": args.checkpoint,
            "debug": debug,
        }

        wandb.log(log_dict)

    # get checkpoint path
    checkpoint_path = args.checkpoint
    assert osp.exists(checkpoint_path), "Checkpoint path does not exist."

    # setup output directory for storing samples

    output_dir = osp.join(
        args.output_dir,
        f"ensemble_prop/{task_name}/{get_datatime()}/flow_nsteps_{nsteps}/",
    )

    if not debug:
        os.makedirs(output_dir, exist_ok=True)

    # load count indices path, indices for what smiles to use
    count_indices_path = args.count_indices
    assert osp.exists(count_indices_path), "Count indices path does not exist."
    log.info(f"Loading count indices from: {count_indices_path}")
    indices, counts = np.load(count_indices_path)
    log.info("Will be generating samples for 100 random molecules.")

    main(
        config=config,
        checkpoint_path=checkpoint_path,
        xtb_path=xtb_path,
        indices=indices,
        counts=counts,
        nsteps=nsteps,
        num_workers=num_workers,
        debug=debug,
        seed=args.seed,
    )
