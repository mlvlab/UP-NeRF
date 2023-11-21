import argparse
import os
import random

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from configs.config import parse_args, save_yaml
from models.nerf_system import NeRFSystem


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(hparams):
    assert (hparams["pose.optimize"] == True) or (
        hparams["pose.optimize"] == False and hparams["pose.c2f"] == None
    ), "if you don't optimize poses, pose.c2f must be None"

    setup_seed(hparams["seed"])
    scene_name = hparams["scene_name"]
    exp_name = hparams["exp_name"]
    save_dir = os.path.join(hparams["out_dir"], scene_name, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    if hparams["resume_ckpt"] is not None:
        resume_ckpt = hparams["resume_ckpt"]
        print(f"Restart training from {resume_ckpt}.")
    elif os.path.isfile(os.path.join(save_dir, "ckpts/last.ckpt")):
        resume_ckpt = os.path.join(save_dir, "ckpts/last.ckpt")
        hparams["resume_ckpt"] = resume_ckpt
        print("Restart training from last checkpoint.")

    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, "ckpts"),
        save_last=True,
        monitor="val/psnr",
        mode="max",
        save_top_k=2,
    )
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [checkpoint_callback, pbar]

    if hparams["debug"]:
        logger = None
    else:
        project_name = f"{hparams['dataset_name']}_pose_optimize"
        logger = WandbLogger(name=exp_name, project=project_name)

    # Need to edit max_steps due to implementation of pytorch-lightining(==1.9.0)
    max_steps = (
        hparams["max_steps"] * 2 if hparams["pose.optimize"] else hparams["max_steps"]
    )

    trainer = Trainer(
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=False,
        val_check_interval=hparams["val.log_interval"],
        devices=hparams["num_gpus"],
        accelerator="auto",
        strategy="ddp" if hparams["num_gpus"] > 1 else None,
        num_sanity_val_steps=1,
        benchmark=True,
        profiler="simple" if hparams["num_gpus"] == 1 else None,
    )

    save_yaml(hparams, os.path.join(save_dir, "config.yaml"))
    trainer.fit(system, ckpt_path=hparams["resume_ckpt"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file.", required=True)
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Modify hparams. Example: train.py resume out_dir TRAIN.BATCH_SIZE 2",
    )

    main(parse_args(parser))
