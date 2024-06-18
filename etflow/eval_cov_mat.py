import argparse
import os

from loguru import logger as log

import wandb
from etflow.commons import load_pkl, save_pkl
from etflow.commons.covmat import CovMatEvaluator, print_covmat_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=str, help="Path to the data file", required=True
    )
    parser.add_argument(
        "--num_workers", "-n", type=int, default=1, help="Number of workers"
    )
    parser.add_argument(
        "--use_alignmol",
        "-a",
        action="store_true",
        default=False,
        help="Use alignmol for matching",
    )
    parser.add_argument(
        "--save_rmsd_matrix",
        "-s",
        action="store_true",
        default=False,
        help="Save rmsd matrix for molecules",
    )

    args = parser.parse_args()

    path = args.path
    os.path.exists(path), f"Path {path} does not exist"
    packed_data_list = load_pkl(path)

    # log on weight and biases
    wandb.init(
        project="Energy-Aware-MCG",
        entity="doms-lab",
        name=f"Evaluation Coverage and Matching: Path {path}",
    )

    wandb.run.log({"Path": path})

    use_alignmol = args.use_alignmol
    wandb.run.log({"Use Alignmol": use_alignmol})

    num_workers = args.num_workers
    log.info(f"Using {num_workers} workers for evaluation...")
    evaluator = CovMatEvaluator(num_workers=num_workers, use_alignmol=args.use_alignmol)
    log.info("Evaluation Started...")
    results, rmsd_results = evaluator(packed_data_list)
    log.info("Evaluation finished...")

    # get dataframe of results
    cov_df, matching_metrics = print_covmat_results(results)

    # log as table
    table = wandb.Table(dataframe=cov_df)
    wandb.run.log({"Coverage Metrics": table})

    if args.save_rmsd_matrix:
        dir_path = os.path.dirname(path)
        rmsd_path = os.path.join(dir_path, "rmsd_results.pkl")
        save_pkl(file_path=rmsd_path, data=rmsd_results)

    # log matching metrics
    wandb.run.log(matching_metrics)
