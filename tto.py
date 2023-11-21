import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

import wandb
from configs.config import get_from_path
from models.nerf_system_optmize import NeRFSystemOptimize


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(hparams):
    setup_seed(hparams["seed"])
    system = NeRFSystemOptimize(hparams)
    scene_name = hparams["scene_name"]
    exp_name = hparams["exp_name"]
    save_dir = os.path.join(hparams["out_dir"], scene_name, exp_name, "tto")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, "ckpts"),
        save_last=False,
        monitor="val/psnr",
        mode="max",
        save_top_k=0,
    )
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [checkpoint_callback, pbar]
    if hparams["pose_optimize"]:
        if hparams["wandb"]:
            wandb_exp_name = "_".join(
                [exp_name, "pose" + str(hparams["optimize_num"]).zfill(2)]
            )
            logger = (
                None
                if hparams["debug"]
                else WandbLogger(
                    name=wandb_exp_name, project=f"{hparams['dataset_name']}_tto"
                )
            )
        else:
            logger = None
        trainer = Trainer(
            max_epochs=50,
            callbacks=callbacks,
            logger=logger,
            enable_model_summary=False,
            devices=hparams["num_gpus"],
            accelerator="auto",
            strategy="ddp" if hparams["num_gpus"] > 1 else None,
            num_sanity_val_steps=1,
            benchmark=True,
            profiler="simple" if hparams["num_gpus"] == 1 else None,
        )
    else:
        if args.wandb:
            if hparams["debug"]:
                logger = None
            else:
                w_exp_name = "_".join(
                    [exp_name, "aemb" + str(hparams["optimize_num"]).zfill(2)]
                )
                project_name = f"{hparams['dataset_name']}_tto"
                logger = WandbLogger(name=w_exp_name, project=project_name)
        else:
            logger = None
        trainer = Trainer(
            max_epochs=20,
            callbacks=callbacks,
            logger=logger,
            enable_model_summary=False,
            devices=hparams["num_gpus"],
            accelerator="auto",
            strategy="ddp" if hparams["num_gpus"] > 1 else None,
            num_sanity_val_steps=1,
            benchmark=True,
            profiler="simple" if hparams["num_gpus"] == 1 else None,
        )
    trainer.fit(system)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir", required=True, type=str, help="Path of result directory."
    )
    parser.add_argument("--ckpt", default="last", type=str, help="Check point name.")
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size.")
    parser.add_argument(
        "--optimize_num", default=-1, type=int, help="Number of test image to optimize."
    )
    parser.add_argument("--wandb", action="store_true", help="Log tto process.")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Modify hparams. Example: train.py resume out_dir TRAIN.BATCH_SIZE 2",
    )
    args = parser.parse_args()

    config_path = os.path.join(args.result_dir, "config.yaml")
    hparams = get_from_path(config_path)
    hparams["ckpt_path"] = os.path.join(args.result_dir, f"ckpts/{args.ckpt}.ckpt")
    hparams["ckpt_path"] = os.path.join(args.result_dir, f"ckpts/{args.ckpt}.ckpt")
    hparams["train.batch_size"] = args.batch_size
    hparams["wandb"] = args.wandb

    if args.optimize_num == -1:  # all test images
        scene_name = hparams["scene_name"]
        tsv = os.path.join(hparams["root_dir"], f"{scene_name}.tsv")
        files = pd.read_csv(tsv, sep="\t")
        test_N = sum(files["split"] == "test")
        optimize_nums = range(test_N)
    else:
        optimize_nums = [args.optimize_num]

    pbar = tqdm(optimize_nums, desc=f"[{1}/{test_N}] Test time optmization.")
    print(f"Start test time optimization of {test_N} test images.")
    for o_n in pbar:
        pbar.set_description(f"[{o_n+1}/{test_N}] Test time optmization.")
        hparams["optimize_num"] = o_n
        hparams["pose_optimize"] = True
        main(hparams)
        wandb.finish()

        hparams["pose_optimize"] = False
        main(hparams)
        wandb.finish()
