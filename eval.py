import argparse
import os
import pickle

import numpy as np
import torch

import utils.camera as camera_utils
import utils.metric as metric_utils
from datasets import dataset_dict


def main(checkpoint):
    hparams = checkpoint["hyper_parameters"]
    se3_refine = checkpoint["state_dict"]["se3_refine.weight"]

    # pose
    dataset = dataset_dict["phototourism"](
        hparams["root_dir"],
        scene_name=hparams["scene_name"],
        feat_dir=None,
        depth_dir=None,
        split="train",
        camera_noise=hparams["pose.noise"],
        img_downscale=hparams["phototourism.img_downscale"],
        use_cache=True,
    )
    noised_poses = torch.stack([dataset.poses_dict[i] for i in dataset.img_ids_train])
    gt_poses = torch.stack(
        [torch.from_numpy(dataset.GT_poses_dict[i]) for i in dataset.img_ids_train]
    )
    pose_refine_ = camera_utils.lie.se3_to_SE3(se3_refine).cpu()
    refine_poses = camera_utils.pose.compose([pose_refine_, noised_poses])

    pose_error, aligned_poses, gt_poses = metric_utils.pose_metric(
        refine_poses, gt_poses
    )
    print("train/pose_R", pose_error["R"].mean() * 180 / np.pi)
    print("train/pose_t", pose_error["t"].mean())

    # novel view synthesis (need to run tto.py)
    root_dir = os.path.join(
        hparams["out_dir"], hparams["scene_name"], hparams["exp_name"], "a_optimize"
    )
    psnr_path = os.path.join(root_dir, "psnr.pkl")
    ssim_path = os.path.join(root_dir, "ssim.pkl")
    lpips_path = os.path.join(root_dir, "lpips.pkl")

    if not os.path.isfile(psnr_path):
        print(f"There is no {psnr_path}.")
        print("You should run tto.py for getting NVS results.")
    else:
        with open(psnr_path, "rb") as f:
            psnr = pickle.load(f)
        with open(ssim_path, "rb") as f:
            ssim = pickle.load(f)
        with open(lpips_path, "rb") as f:
            lpips = pickle.load(f)
        psnr = [v.item() for v in psnr.values()]
        ssim = [v.item() for v in ssim.values()]
        lpips = [v.item() for v in lpips.values()]
        print("PSNR: \t", np.mean(psnr))
        print("SSIM: \t", np.mean(ssim))
        print("LPIPS: \t", np.mean(lpips))
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir", required=True, type=str, help="Path of result directory."
    )
    parser.add_argument("--ckpt", default="last", type=str, help="Check point name.")
    args = parser.parse_args()

    ckpt_path = os.path.join(args.result_dir, f"ckpts/{args.ckpt}.ckpt")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    main(checkpoint)
