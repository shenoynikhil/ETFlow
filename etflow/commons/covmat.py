from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from typing import Callable, List

import datamol as dm
import numpy as np
import pandas as pd
import torch
from datamol.types import Mol
from loguru import logger as log
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdchem import Conformer
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Geometry import Point3D
from tqdm import tqdm


def build_conformer(pos):
    if isinstance(pos, torch.Tensor) or isinstance(pos, np.ndarray):
        pos = pos.tolist()

    conformer = Conformer()

    for i, atom_pos in enumerate(pos):
        conformer.SetAtomPosition(i, Point3D(*atom_pos))

    return conformer


def set_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = deepcopy(rdkit_mol)
    conformer = build_conformer(pos)
    mol.AddConformer(conformer)
    return mol


def get_best_rmsd(probe, ref, use_alignmol=False):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)

    try:
        if use_alignmol:
            return MA.AlignMol(probe, ref)
        else:
            rmsd = MA.GetBestRMS(probe, ref)
    except:  # noqa
        rmsd = np.nan

    return rmsd


def get_rmsd(ref_mol: Mol, gen_mols: List[Mol], useFF=False, use_alignmol=False):
    num_gen = len(gen_mols)
    rmsd_vals = []
    for i in range(num_gen):
        gen_mol = gen_mols[i]
        if useFF:
            # print('Applying FF on generated molecules...')
            MMFFOptimizeMolecule(gen_mol)
        rmsd_vals.append(get_best_rmsd(gen_mol, ref_mol, use_alignmol=use_alignmol))

    return rmsd_vals


def calc_performance_stats(rmsd_array, threshold):
    coverage_recall = np.mean(
        np.nanmin(rmsd_array, axis=1, keepdims=True) < threshold, axis=0
    )
    amr_recall = np.mean(np.nanmin(rmsd_array, axis=1))
    coverage_precision = np.mean(
        np.nanmin(rmsd_array, axis=0, keepdims=True) < np.expand_dims(threshold, 1),
        axis=1,
    )
    amr_precision = np.mean(np.nanmin(rmsd_array, axis=0))

    return coverage_recall, amr_recall, coverage_precision, amr_precision


def worker_fn(job, useFF=False, use_alignmol=False):
    smi, i_true, ref_mol, gen_mols = job
    rmsd_vals = get_rmsd(ref_mol, gen_mols, useFF=useFF, use_alignmol=use_alignmol)
    return smi, i_true, rmsd_vals


class CovMatEvaluator(object):
    """Coverage Recall Metrics Calculation for GEOM-Dataset"""

    def __init__(
        self,
        num_workers: int = 8,
        use_force_field: bool = False,
        use_alignmol: bool = False,
        thresholds: np.ndarray = np.arange(0.05, 3.05, 0.05),
        ratio: int = 2,
        filter_disconnected: bool = True,
        print_fn: Callable = print,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.use_force_field = use_force_field
        self.use_alignmol = use_alignmol
        self.thresholds = np.array(thresholds).flatten()

        self.ratio = ratio
        self.filter_disconnected = filter_disconnected

        self.print_fn = print_fn

    def __call__(self, packed_data_list, start_idx=0):
        rmsd_results = {}
        jobs = []
        true_mols = {}
        gen_mols = {}
        for data in packed_data_list:
            if "pos_gen" not in data or "pos_ref" not in data:
                log.info("skipping due to missing pos_gen or pos_ref")
                continue
            if self.filter_disconnected and ("." in data["smiles"]):
                log.info("skipping due to disconnected molecule")
                continue

            num_atoms = data["pos_gen"].shape[1]
            if isinstance(data["pos_gen"], torch.Tensor):
                data["pos_gen"] = data["pos_gen"].cpu().numpy()

            smiles = data["smiles"]
            mol = dm.to_mol(smiles, remove_hs=False, ordered=True)
            data["pos_ref"] = data["pos_ref"].reshape(-1, num_atoms, 3)
            data["pos_gen"] = data["pos_gen"].reshape(-1, num_atoms, 3)

            num_true = data["pos_ref"].shape[0]
            num_gen = num_true * self.ratio
            if data["pos_gen"].shape[0] < num_gen:
                log.info("skipping due to insufficient number of generated conformers")
                continue
            data["pos_gen"] = data["pos_gen"][:num_gen]

            true_mols[smiles] = [
                set_rdmol_positions(mol, data["pos_ref"][i]) for i in range(num_true)
            ]
            gen_mols[smiles] = [
                set_rdmol_positions(mol, data["pos_gen"][i]) for i in range(num_gen)
            ]

            rmsd_results[smiles] = {
                "n_true": num_true,
                "n_model": num_gen,
                "rmsd": np.nan * np.ones((num_true, num_gen)),
            }
            for i in range(num_true):
                jobs.append((smiles, i, true_mols[smiles][i], gen_mols[smiles]))

        # remove packed_data_list from memory
        del packed_data_list

        def populate_results(res):
            smi, i_true, rmsd_vals = res
            rmsd_results[smi]["rmsd"][i_true] = rmsd_vals

        if self.num_workers > 1:
            p = Pool(self.num_workers)
            map_fn = p.imap_unordered
            p.__enter__()
        else:
            map_fn = map

        fn = partial(
            worker_fn, useFF=self.use_force_field, use_alignmol=self.use_alignmol
        )
        for res in tqdm(map_fn(fn, jobs), total=len(jobs)):
            populate_results(res)

        if self.num_workers > 1:
            p.__exit__(None, None, None)

        stats = []
        for res in rmsd_results.values():
            stats_ = calc_performance_stats(res["rmsd"], self.thresholds)
            stats.append(stats_)
        coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)

        results = {
            "CoverageR": np.array(coverage_recall),  # (num_mols, num_threshold)
            "MatchingR": np.array(amr_recall),  # (num_mols)
            "thresholds": self.thresholds,
            "CoverageP": np.array(coverage_precision),  # (num_mols, num_threshold)
            "MatchingP": np.array(amr_precision),  # (num_mols)
        }
        # print_conformation_eval_results(results)
        return results, rmsd_results


def print_covmat_results(results, print_fn=print):
    df = pd.DataFrame.from_dict(
        {
            "Threshold": results["thresholds"],
            "COV-R_mean": np.mean(results["CoverageR"], 0),
            "COV-R_median": np.median(results["CoverageR"], 0),
            "COV-P_mean": np.mean(results["CoverageP"], 0),
            "COV-P_median": np.median(results["CoverageP"], 0),
        }
    )
    matching_metrics = {
        "MAT-R_mean": np.mean(results["MatchingR"]),
        "MAT-R_median": np.median(results["MatchingR"]),
        "MAT-P_mean": np.mean(results["MatchingP"]),
        "MAT-P_median": np.median(results["MatchingP"]),
    }
    return df, matching_metrics
