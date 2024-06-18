import argparse
import os.path as osp

import torch
from lightning.pytorch import seed_everything
from loguru import logger as log

from etflow.utils import (
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_logger,
    instantiate_model,
    instantiate_trainer,
    log_hyperparameters,
    read_yaml,
    setup_log_dir,
)

torch.set_float32_matmul_precision("high")


def run(config: dict) -> None:
    # check if debug mode
    debug = config.get("debug", False)

    # seed everything for reproducibility
    seed_everything(config.get("seed", 42))

    # task name for logger, if not provided use default
    task_name = config.get("task_name", None)
    if debug:
        task_name = "debug-run"
    assert task_name is not None, "Task name not provided"

    # instantiate logger (skip if debug mode)
    logger = None
    if config.get("logger") is not None:
        logger = instantiate_logger(
            config.get("logger"),
            config.get("logger_args"),
            task_name=task_name,
            debug_mode=debug,
            no_logger=config.get("no_logger", False),
        )

    # setup log directory
    setup_log_dir(task_name)

    # instantiate datamodule
    datamodule = instantiate_datamodule(config["datamodule"], config["datamodule_args"])

    # instantiate model
    model = instantiate_model(config["model"], config["model_args"])
    pretrained_ckpt = config.get("pretrained_ckpt", None)
    if pretrained_ckpt is not None:
        assert osp.exists(
            pretrained_ckpt
        ), f"Pretrained checkpoint {pretrained_ckpt} not found!"
        state_dict = torch.load(pretrained_ckpt, map_location=model.device)[
            "state_dict"
        ]
        model.load_state_dict(state_dict)
        log.info(f"Loaded pretrained model from checkpoint: {pretrained_ckpt}")

    # instantiate callbacks
    callbacks = instantiate_callbacks(config["callbacks"])

    # instantiate trainer
    trainer = instantiate_trainer(
        config["trainer"],
        config["trainer_args"],
        logger=logger,
        callbacks=callbacks,
        debug=debug,
    )

    # log config
    log_hyperparameters({"cfg": config, "model": model, "trainer": trainer})

    # start training
    resume_ckpt_path = config.get("ckpt_path", None)
    if resume_ckpt_path is not None:
        print(f"Resuming training from checkpoint: {resume_ckpt_path}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    # read config path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--no_logger", "-n", action="store_true")
    args = parser.parse_args()

    # read config
    osp.exists(args.config), f"Config file {args.config} not found"
    config = read_yaml(args.config)

    # update config with debug mode
    config["debug"] = args.debug
    config["no_logger"] = args.no_logger
    run(config)
