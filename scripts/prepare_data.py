"""
Preprocessing Script for GEOM dataset
=====================================

Usage
```bash
# make sure $DATA_DIR is set
# processed files will be saved at $DATA_DIR/processed
python scripts/prepare_data.py -p /path/to/geom/rdkit-raw-folder
```
"""

import argparse
import os
import os.path as osp
from typing import Dict

import datamol as dm
import numpy as np
from loguru import logger as log
from tqdm import tqdm

from etflow.commons import (
    get_atomic_number_and_charge,
    get_base_data_dir,
    load_json,
    load_pkl,
)

TOTAL_NUM_MOLS = 437580
TOTAL_ATOMS_UNIQUE_MOLS = 15904498  # only counting once per molecule
TOTAL_NUM_ATOMS = 1656320500  # counting for all conformers
TOTAL_NUM_CONFORMERS = 33078483


def read_mol(
    mol_id: str,
    mol_dict,
    base_path: str,
    partition: str,
    pos_start_idx: int,
) -> Dict[str, np.ndarray]:
    try:
        d = load_pkl(osp.join(base_path, mol_dict["pickle_path"]))
        confs = d["conformers"]
        mols = [conf["rd_mol"] for conf in confs]

        # maps the atom index to atoms in smiles string
        # replicates the exact mol structure i.e.
        # atom index and edge index
        smiles_ = dm.to_smiles(
            mols[0],
            canonical=False,
            explicit_hs=True,
            with_atom_indices=True,
            isomeric=True,
        )
        smiles = [smiles_] * len(confs)

        # atom specific information, positions, atomic numbers and charges
        positions: np.ndarray = np.concatenate(
            [cf["rd_mol"].GetConformer().GetPositions() for cf in confs], 0
        )
        num_atoms, num_confs = mols[0].GetNumAtoms(), len(mols)
        x: np.ndarray = np.concatenate(
            [get_atomic_number_and_charge(mol) for mol in mols]
        )
        pos_index_range: np.ndarray = (
            np.arange(
                pos_start_idx, pos_start_idx + num_atoms * num_confs + 1, num_atoms
            )
            .repeat(2)[1:-1]
            .reshape(-1, 2)
        )

        # # energy
        energy = [cf["totalenergy"] for cf in confs]

        return dict(
            # atom based
            atomic_inputs=np.concatenate([x, positions], axis=-1, dtype=np.float32),
            pos_index_range=pos_index_range,
            num_atoms=num_atoms,
            # energy
            energy=energy,
            # string based
            smiles=smiles,
            subset=partition,
            num_confs=num_confs,
        )

    except Exception as e:
        print(f"Skipping: {mol_id} due to {e}")
        return None


def preprocess(raw_path: str, dest_folder_path: str) -> None:
    log.info(f"Reading files from {raw_path}")
    partitions = ["qm9", "drugs"]
    processed_list = []
    total_confs, total_atoms, total_atoms_mols, total_edges, total_mols = 0, 0, 0, 0, 0

    memmap_dict = init_memmap(dest_folder_path)

    for partition in partitions:
        mols = load_json(osp.join(raw_path, f"summary_{partition}.json"))

        for _, (mol_id, mol_dict) in tqdm(
            enumerate(mols.items()),
            total=len(mols),
            desc=f"Processing molecules of {partition}",
        ):
            res = read_mol(
                mol_id,
                mol_dict,
                raw_path,
                partition,
                total_atoms,
            )

            if res is None:
                continue

            num_atoms = res.pop("num_atoms")
            smiles = res.pop("smiles")
            subset = res.pop("subset")
            num_confs = res.pop("num_confs")
            num_atoms_confs = num_atoms * num_confs

            index_dict = {
                "atomic_inputs": [total_atoms, total_atoms + num_atoms_confs],
                "pos_index_range": [total_confs, total_confs + num_confs],
                "energy": [total_confs, total_confs + num_confs],
            }

            for key in res:
                index_range = index_dict[key]
                begin_idx, end_idx = index_range
                if key == "edge_index":
                    memmap_dict[key][:, begin_idx:end_idx] = res[key]
                else:
                    memmap_dict[key][begin_idx:end_idx] = res[key]

            total_mols += 1
            total_atoms += num_atoms_confs
            total_confs += num_confs
            total_atoms_mols += num_atoms

            pkl_res = {"smiles": smiles, "subset": subset, "num_confs": num_confs}

            processed_list.append(pkl_res)

    log.info(
        f"Processed: {total_mols} molecules, {total_atoms} atoms, "
        f"{total_atoms_mols} atoms (unique mols), {total_edges} edges, "
        f"{total_confs} conformers."
    )

    # save the memmaps
    for key in memmap_dict:
        memmap_dict[key].flush()

    # save smiles and subsets
    save(processed_list, dest_folder_path)


def init_memmap(dest_folder_path):
    log.info(f"Saving to {dest_folder_path}")

    atomic_inputs_path = osp.join(dest_folder_path, "atomic_inputs.memmap")
    pos_index_range_path = osp.join(dest_folder_path, "pos_index_range.memmap")
    energy_path = osp.join(dest_folder_path, "energy.memmap")

    return {
        "atomic_inputs": np.memmap(
            atomic_inputs_path, mode="w+", dtype=np.float32, shape=(TOTAL_NUM_ATOMS, 5)
        ),
        "pos_index_range": np.memmap(
            pos_index_range_path,
            mode="w+",
            dtype=np.int32,
            shape=(TOTAL_NUM_CONFORMERS, 2),
        ),
        "energy": np.memmap(
            energy_path, mode="w+", dtype=np.float32, shape=(TOTAL_NUM_CONFORMERS,)
        ),
    }


def save(
    processed_list,
    dest_folder_path: str,
):
    # save
    log.info("Saving Smiles and Subset now!")

    # dictionary for constant time lookup
    smiles_dict = {}
    subset_dict = {}

    smiles_inv_indices = []
    subset_inv_indices = []
    smiles_index = 0
    subset_index = 0
    for x in tqdm(processed_list, desc="Processing Smiles and Subset"):
        # filter the processed list removing NaNs
        if x is None:
            continue

        smiles_list = x["smiles"]
        subset = x["subset"]
        n = x["num_confs"]

        for smiles in smiles_list:
            if smiles not in smiles_dict:
                smiles_dict[smiles] = smiles_index
                smiles_index += 1

        if subset not in subset_dict:
            subset_dict[subset] = subset_index
            subset_index += 1

        smiles_inv_indices.append(
            [smiles_dict[smi] for smi in smiles_list]
        )  # different index for each conformer
        subset_inv_indices.append([subset_dict[subset]] * n)  # same index for subset

    smiles = np.asarray(list(smiles_dict.keys()))
    subset = np.asarray(list(subset_dict.keys()))
    smiles_inv_indices = np.hstack(smiles_inv_indices).astype(int)
    subset_inv_indices = np.hstack(subset_inv_indices).astype(int)

    file_path = osp.join(dest_folder_path, "smiles.npz")
    np.savez_compressed(file_path, uniques=smiles, inv_indices=smiles_inv_indices)

    file_path = osp.join(dest_folder_path, "subset.npz")
    np.savez_compressed(file_path, uniques=subset, inv_indices=subset_inv_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to the geom dataset rdkit folder",
    )
    # destination path to store
    args = parser.parse_args()

    # get path to raw file
    path = args.path
    assert osp.exists(path), f"Path {path} not found"

    # get distanation path
    dest = osp.join(get_base_data_dir(), "processed")
    os.makedirs(dest, exist_ok=True)
    log.info(f"Processed files will be saved at destination path: {dest}")

    # preprocess
    preprocess(path, dest)
