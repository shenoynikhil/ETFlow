"""
Script to process GEOM dataset into individual PyG data files
===========================================================

Usage:
# Set the DATA_DIR environment variable before running this script
python prepare_data.py -p /path/to/geom/rdkit-raw-folder
"""

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Optional

import datamol as dm
import numpy as np
import torch
from loguru import logger as log
from torch_geometric.data import Data
from tqdm import tqdm

from etflow.commons import get_base_data_dir, load_pkl


def check_disconnected_components(mol):
    """Check for disconnected components using Union-Find algorithm."""
    # Initialize parent array for union-find
    n_nodes = mol.GetNumAtoms()
    parent = list(range(n_nodes))

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Add reverse edge for undirected graph
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # Process all edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        union(src, dst)

    # Count unique components
    components = {}
    for node in range(n_nodes):
        root = find(node)
        if root not in components:
            components[root] = []
        components[root].append(node)

    return list(components.values())


def process_test_mol(mols: list[dm.Mol], partition: str):
    """Process a single test molecule into a relevant PyG data object."""
    try:
        mol = mols[0]
        smiles = dm.to_smiles(
            mol,
            canonical=False,
            explicit_hs=True,
            with_atom_indices=True,
            isomeric=True,
        )

        atomic_numbers = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long
        )
        atomic_charges = torch.tensor(
            [atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=torch.long
        )

        # get conformer positions from mol.GetConformer().GetPositions()
        pos = torch.tensor([mol.GetConformer().GetPositions() for mol in mols]).float()
        data = Data(
            atomic_numbers=atomic_numbers,
            charges=atomic_charges,
            pos=pos,
            smiles=smiles,
            subset=partition,
        )
        return data
    except Exception as e:
        log.warning(f"Skipping: {smiles} due to {e}")
        return None


def process_mol(
    pkl_path: Path,
    partition: str,
    split: str,
    keep_top_n: int = 30,
) -> Optional[Data]:
    """Process a single molecule with all its conformers into one PyG Data object."""
    try:
        # Load molecule data
        mol_dict = load_pkl(pkl_path)
        confs = mol_dict["conformers"]
        mols = [conf["rd_mol"] for conf in confs]

        # Get SMILES with atom indices (use first mol as they're all same)
        smiles = dm.to_smiles(
            mols[0],
            canonical=False,
            explicit_hs=True,
            with_atom_indices=True,
            isomeric=True,
        )

        # Get chemical information (same for all conformers)
        mol = mols[0]
        atomic_numbers = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long
        )
        atomic_charges = torch.tensor(
            [atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=torch.long
        )

        # Sort conformers by Boltzmann weight if train/val split
        if split in ["train", "val"]:
            confs = sorted(confs, key=lambda x: x["boltzmannweight"], reverse=True)
            confs = confs[:keep_top_n]  # Keep top 30 conformers

        # Collect conformer positions and energies
        positions = []
        energies = []
        boltzmann_weights = []  # Also store the weights

        for conf in confs:
            mol = conf["rd_mol"]
            pos = torch.from_numpy(mol.GetConformer().GetPositions()).float()
            energy = torch.tensor([conf["totalenergy"]]).float()
            weight = torch.tensor([conf["boltzmannweight"]]).float()

            positions.append(pos)
            energies.append(energy)
            boltzmann_weights.append(weight)

        # Stack conformer data
        positions = torch.stack(positions)  # [num_conformers, num_atoms, 3]
        energies = torch.stack(energies)  # [num_conformers, 1]
        boltzmann_weights = torch.stack(boltzmann_weights)  # [num_conformers, 1]

        # Check for disconnected components
        components = check_disconnected_components(mol)
        if len(components) > 1:
            log.warning(f"Skipping {smiles} due to disconnected components")
            return None

        # Create single PyG Data object with all conformers
        data = Data(
            atomic_numbers=atomic_numbers,  # [num_atoms]
            charges=atomic_charges,  # [num_atoms]
            pos=positions,  # [num_conformers, num_atoms, 3]
            energy=energies,  # [num_conformers, 1]
            smiles=smiles,
            subset=partition,
        )

        return data

    except Exception as e:
        log.warning(f"Skipping due to {e}")
        return None


def main(raw_path: Path, output_dir: Path, data_dir: Path) -> None:
    log.info(f"Reading files from {raw_path}")

    # Create output directories for each partition and split
    partitions = ["qm9", "drugs"]
    for partition in partitions:
        for split in ["train", "val", "test"]:
            (output_dir / partition / split).mkdir(parents=True, exist_ok=True)

    # Statistics tracking
    stats = {
        partition: {
            split: {"mols": 0, "confs": 0} for split in ["train", "val", "test"]
        }
        for partition in partitions
    }
    skipped_mols = 0

    # Process each partition
    for partition in partitions:
        log.info(f"Processing {partition} partition")

        # Load molecule summary
        all_pkl_files = list((raw_path / partition).glob("*"))
        splits = np.load(data_dir / partition.upper() / "split.npy", allow_pickle=True)
        train_split_indices = splits[0]
        val_split_indices = splits[1]

        # train pickle paths
        train_pkl_paths = [all_pkl_files[i] for i in train_split_indices]
        val_pkl_paths = [all_pkl_files[i] for i in val_split_indices]

        # Process each molecule
        for split, pkl_paths in zip(["train", "val"], [train_pkl_paths, val_pkl_paths]):
            # Process each pickle file in the current split
            for pkl_path in tqdm(pkl_paths, desc=f"Processing {partition} {split}"):
                # Extract molecule ID from path
                mol_id = pkl_path.stem

                # skip if file already exists
                if (output_dir / partition / split / f"{mol_id}.pt").exists():
                    continue

                # Process molecule with all its conformers
                data = process_mol(pkl_path, partition, split)

                if data is None:
                    skipped_mols += 1
                    continue

                # Save molecule data in appropriate split directory
                save_path = output_dir / partition / split / f"{mol_id}.pt"

                # Save the PyG Data object
                torch.save(data, save_path)

                # Update statistics
                stats[partition][split]["mols"] += 1

        # save test set data objects
        log.info(f"Processing test molecules for {partition}")
        test_mols: Dict[str, list[dm.Mol]] = load_pkl(
            data_dir / partition.upper() / "test_mols.pkl"
        )
        for test_mol_id, test_mol in test_mols.items():
            data = process_test_mol(test_mol, partition)
            if data is None:
                skipped_mols += 1
                continue

            safe_test_mol_id = hashlib.md5(test_mol_id.encode()).hexdigest()
            save_path = output_dir / partition / "test" / f"{safe_test_mol_id}.pt"
            torch.save(data, save_path)

    # Log statistics
    log.info("Processing complete:")
    for partition in partitions:
        log.info(f"\n{partition.upper()} statistics:")
        for split in ["train", "val", "test"]:
            split_stats = stats[partition][split]
            log.info(
                f"- {split}: {split_stats['mols']} molecules, "
                f"{split_stats['confs']} conformers"
            )
    log.info(f"Skipped molecules: {skipped_mols}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        required=True,
        help="Path to the geom dataset rdkit folder",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        required=False,
        help="Directory to save processed files",
    )
    parser.add_argument(
        "--remove_disconnected_components",
    )

    args = parser.parse_args()

    # Verify input paths exist
    assert args.path.exists(), f"Path {args.path} not found"
    data_dir = Path(get_base_data_dir())
    # Verify the the splits are present
    for partition in ["QM9", "DRUGS"]:
        assert (
            data_dir / partition / "split.npy"
        ).exists(), f"Split not found in {partition}"
        assert (
            data_dir / partition / "test_mols.pkl"
        ).exists(), f"Test split mols not found in {partition}"

    # Create output directory as output_dir/processed
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(get_base_data_dir()) / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process the data
    main(args.path, output_dir, data_dir)
