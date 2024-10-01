"""
Evaluation CSV Preparation
--------------------------
This script prepares an evaluation CSV for the geom dataset. This CSV maps each conformer to the
correct SMILES string (instead of using 1 SMILES string for all conformers).

To prepare the evaluation CSV, we need to run the following command,
```bash
# make sure $DATA_DIR is set
# will save the evaluation csv at $DATA_DIR/processed/geom.csv
python scripts/prepare_eval_csv.py --path /path/to/geom/rdkit-raw-folder
```
"""

import argparse
import os
import os.path as osp

import datamol as dm
import pandas as pd
from loguru import logger as log
from tqdm import tqdm

from etflow.commons import get_base_data_dir, load_json, load_pkl


def get_smiles(mol) -> str:
    return dm.to_smiles(
        mol,
        canonical=False,
        explicit_hs=True,
        with_atom_indices=True,
        isomeric=True,
    )


def get_data(mol_id: str, mol_dict: dict, partition: str, base_path: str):
    """Given a molecule id and its dictiona"""
    try:
        d = load_pkl(osp.join(base_path, mol_dict["pickle_path"]))
        confs = d["conformers"]
        update = [
            {
                # "mol_key": mol_id,
                "smiles": get_smiles(conf["rd_mol"]),
                "boltzmannweight": conf["boltzmannweight"],
                "partition": partition,
                "energy": conf["totalenergy"],
            }
            for conf in confs
        ]
        return update
    except Exception as e:
        log.warning(f"Error in {mol_id}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to the geom dataset rdkit folder",
    )
    args = parser.parse_args()

    # geom raw data path
    geom_raw_data_path = args.path
    assert osp.exists(geom_raw_data_path), "Path does not exist"

    # destination path for csv
    os.makedirs(osp.join(get_base_data_dir(), "processed"), exist_ok=True)
    dest_csv_path = osp.join(get_base_data_dir(), "processed/geom.csv")

    drugs_data = load_json(osp.join(geom_raw_data_path, "summary_drugs.json"))
    qm9_data = load_json(osp.join(geom_raw_data_path, "summary_qm9.json"))

    processed_data = []
    base_path = "/mnt/ps/home/CORP/nikhil.shenoy/rdkit_folder/"

    log.info("Processing data for qm9")
    for mol_id, mol_dict in tqdm(qm9_data.items()):
        update = get_data(mol_id, mol_dict, "qm9", base_path)

        if update is None:
            continue

        processed_data.extend(update)

    log.info("Processing data for drugs")
    for mol_id, mol_dict in tqdm(drugs_data.items()):
        update = get_data(mol_id, mol_dict, "drugs", base_path)

        if update is None:
            continue

        processed_data.extend(update)

    df = pd.DataFrame(processed_data)
    log.info(f"Saving Evaluation CSV at {dest_csv_path}")
    df.to_csv(dest_csv_path, index=False)
