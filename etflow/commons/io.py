import json
import os
import pickle

import numpy as np

CACHE_DIR = os.environ.get("CACHE_DIR", "~/.cache/data")


def load_memmap(path, dtype):
    return np.memmap(path, dtype=dtype, mode="r")


def load_pkl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pkl(file_path: str, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_json(path):
    """Loads json file"""
    with open(path, "r") as fp:  # Unpickling
        return json.load(fp)


def load_npz(path):
    data_dict = np.load(path)
    uniques = data_dict["uniques"]
    inv_indices = data_dict["inv_indices"]
    return uniques, inv_indices


def save_memmap(data, path, shape, dtype):
    f = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    f[:] = data
    f.flush()


def get_local_cache() -> str:
    """
    Returns the local cache directory. It creates it if it does not exist.

    Returns:
        str: path to the local cache directory
    """
    cache_dir = os.path.expanduser(os.path.expandvars(CACHE_DIR))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_base_data_dir() -> str:
    # get the environment variable DATA_DIR
    data_dir = os.environ.get("DATA_DIR", None)

    if data_dir is None:
        raise ValueError("DATA_DIR environment variable is not set.")

    return data_dir
