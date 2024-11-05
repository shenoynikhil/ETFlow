import os
from datetime import datetime
from typing import List, Optional

import torch
import yaml
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger as log
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, ReduceLROnPlateau

from etflow.data.datamodule import BaseDataModule
from etflow.data.dataset import EuclideanDataset
from etflow.models.model import BaseFlow
from etflow.schedulers import CosineAnnealingWarmupRestarts


def set_to_tensor(value: float):
    if isinstance(value, float):
        return torch.tensor([value], dtype=torch.float32)
    # return as it is if already tensor
    return value


def get_log_dir():
    """
    Check if LOG_DIR is env variable is set
    else use logs/ directory
    """
    log_dir = os.environ.get("LOG_DIR")
    if log_dir is None:
        log_dir = "logs/"

    return log_dir


def setup_log_dir(task_name):
    """Sets log directory for a given task
    and then moves to that directory
    """
    log_dir = get_log_dir()
    # use time to create unique log diedge_indexrectory
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_dir, task_name, "runs", f"{run_name}")
    log.info(f"Log directory: {log_dir}")

    # create log directory
    os.makedirs(log_dir, exist_ok=True)
    os.chdir(log_dir)


def read_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_partial_weights(model, ckpt_path, target: str = None):
    """
    target: name to match from state_dict
        e.g.1 "encoder"(in case of vae-encoder)
        e.g.2 "network"(in case of pretraiend torchmdnet energy predictor)
    """
    ckpt = torch.load(ckpt_path)["state_dict"]
    ckpt_keys = None
    if target is not None:
        ckpt_keys = [k for k in ckpt.keys() if target in k]
    model_weights = model.state_dict()

    # Filter out unnecessary keys
    matching_keys = {
        k1: k2 for k1 in ckpt_keys for k2 in model_weights.keys() if k2 in k1
    }
    weight_dict = {matching_keys[k]: ckpt[k] for k in matching_keys.keys()}

    # Update model's state_dict with the filtered checkpoint
    model_weights.update(weight_dict)
    model.load_state_dict(model_weights)
    return weight_dict


def instantiate_optimizer(
    optimizer_type: str, optimizer_args: dict, networks: List[torch.nn.Module]
) -> Optimizer:
    # collect parameters
    parameters = []
    for network in networks:
        parameters += list(network.parameters())

    # initialize optimizer
    if optimizer_type == "Adam":
        log.info("Initializing Adam as optimizer")
        optimizer = Adam(parameters, **optimizer_args)
    elif optimizer_type == "AdamW":
        log.info("Initializing AdamW as optimizer")
        optimizer = AdamW(parameters, **optimizer_args)
    else:
        raise NotImplementedError

    return optimizer


def instantiate_scheduler(
    lr_scheduler_type: str,
    lr_scheduler_args: dict,
    optimizer: Optimizer,
) -> Optional[LRScheduler]:
    if lr_scheduler_type is None:
        return None

    if lr_scheduler_type == "CosineAnnealingLR":
        log.info("Initializing CosineAnnealingLR as lr_scheduler")
        lr_scheduler = CosineAnnealingLR(optimizer, **lr_scheduler_args)
    elif lr_scheduler_type == "CosineAnnealingWarmupRestarts":
        log.info("Initializing CosineAnnealingWarmupRestarts as lr_scheduler")
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, **lr_scheduler_args)
    elif lr_scheduler_type == "ReduceLROnPlateau":
        log.info("Initializing ReduceLROnPlateau as lr_scheduler")
        lr_scheduler = ReduceLROnPlateau(optimizer, **lr_scheduler_args)
    else:
        raise NotImplementedError(f"Scheduler {lr_scheduler_type} not implemented yet.")

    return lr_scheduler


def instantiate_model(
    model_type: str, model_args: dict, stats: Optional[dict] = None
) -> LightningModule:
    if model_type == "BaseFlow":
        log.info(f"Loading BaseFlow with args: {model_args}")
        return BaseFlow(**model_args)

    raise NotImplementedError


def instantiate_dataset(dataset_type: str, dataset_args: dict) -> LightningDataModule:
    if dataset_type == "EuclideanDataset":
        dataset = EuclideanDataset(**dataset_args)
    else:
        raise NotImplementedError

    return dataset


def instantiate_datamodule(
    datamodule_type: str, datamodule_args: dict
) -> BaseDataModule:
    dataset = instantiate_dataset(
        datamodule_args["dataset"], datamodule_args["dataset_args"]
    )
    dataloader_args = datamodule_args["dataloader_args"]
    if datamodule_type == "BaseDataModule":
        datamodule = BaseDataModule(
            dataset=dataset,
            dataloader_args=dataloader_args,
            train_indices_path=datamodule_args.get("train_indices_path", None),
            val_indices_path=datamodule_args.get("val_indices_path", None),
        )
    else:
        raise NotImplementedError

    return datamodule


def instantiate_trainer(
    trainer_type: str,
    trainer_args: dict,
    logger: Logger,
    callbacks: List[Callback],
    debug: bool,
) -> Trainer:
    if debug:
        trainer_args["fast_dev_run"] = 1000
        trainer_args["devices"] = 1  # check on single GPU
        trainer_args["strategy"] = "auto"  # auto select strategy

    if trainer_type == "Trainer":
        trainer = Trainer(**trainer_args, logger=logger, callbacks=callbacks)
    else:
        raise NotImplementedError

    return trainer


def instantiate_logger(
    logger_type: str,
    logger_args: dict,
    task_name: str,
    debug_mode: bool = False,
    no_logger: bool = False,
) -> Logger:
    if debug_mode or no_logger:
        return None

    if logger_type == "WandbLogger":
        if "name" in logger_args:
            del logger_args["name"]  # name is set by task_name
        logger = WandbLogger(**logger_args, name=task_name)
    else:
        raise NotImplementedError

    return logger


def instantiate_callbacks(callbacks: list) -> List[Callback]:
    final_callbacks = []
    for callback_dict in callbacks:
        if callback_dict["callback"] == "ModelCheckpoint":
            final_callbacks.append(ModelCheckpoint(**callback_dict["callback_args"]))
        elif callback_dict["callback"] == "EarlyStopping":
            final_callbacks.append(EarlyStopping(**callback_dict["callback_args"]))
        elif callback_dict["callback"] == "LearningRateMonitor":
            final_callbacks.append(
                LearningRateMonitor(**callback_dict["callback_args"])
            )
        else:
            raise NotImplementedError

    return final_callbacks


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]
    hparams["model_args"] = cfg["model_args"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["datamodule_args"] = cfg["datamodule_args"]
    hparams["trainer"] = cfg["trainer"]
    hparams["trainer_args"] = cfg["trainer_args"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
