import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils.camera as camera_utils
import utils.metric as metric_utils
from datasets import dataset_dict
from models.nerf_system import NeRFSystem
from models.rendering import *
from utils import load_ckpt
from utils.optim import get_learning_rate, get_optimizer
from utils.ray import get_rays


class NeRFSystemOptimize(NeRFSystem):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.best_psnr = 0
        self.best_ssim = 0
        self.best_lpips = 100
        self.scene = hparams["scene_name"]
        exp_name = hparams["exp_name"]
        self.save_root = os.path.join(
            self.hparams["out_dir"], self.scene, exp_name, "a_optimize"
        )
        self.pose_save_dir = os.path.join(self.save_root, "optimized_pose")
        if self.hparams["pose_optimize"]:
            self.save_dir = os.path.join(self.save_root, "optimized_pose")
        else:
            self.save_dir = os.path.join(self.save_root, "optimized_emb_a")
            self.psnr_path = os.path.join(self.save_root, "psnr.pkl")
            self.ssim_path = os.path.join(self.save_root, "ssim.pkl")
            self.lpips_path = os.path.join(self.save_root, "lpips.pkl")

            self.best_psnr_dict = {}
            self.best_ssim_dict = {}
            self.best_lpips_dict = {}
        os.makedirs(self.save_dir, exist_ok=True)

    def setup(self, stage):
        self.dataset_setup()
        self.model_setup()

    def configure_optimizers(self):
        optimizer = []
        if self.hparams["pose_optimize"]:
            self.optimizer = get_optimizer(
                self.hparams["optimizer.type"], 5e-3, self.models_to_train
            )
            optimizer += [self.optimizer]

            self.optimizer_pose = get_optimizer(
                self.hparams["optimizer_pose.type"], 1e-4, self.se3_refine
            )
            optimizer += [self.optimizer_pose]
        else:
            self.optimizer = get_optimizer("adamw", 1e-1, self.models_to_train)
            optimizer += [self.optimizer]

        return optimizer

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=8,
            batch_size=self.hparams["train.batch_size"],
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,  # validate one image (H*W rays) at a time
            pin_memory=True,
        )

    def forward(self, rays, img_idx, train=True):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        chunk = B if train else self.hparams["val.chunk_size"]
        for i in range(0, B, chunk):
            rendered_ray_chunks = render_rays(
                models=self.models,
                embeddings=self.embeddings,
                rays=rays[i : i + chunk],
                img_idx=img_idx[i : i + chunk],
                sched_mult=1.0,
                sched_phase=2,
                N_samples=self.hparams["nerf.N_samples"],
                use_disp=self.hparams["nerf.use_disp"],
                perturb=self.hparams["nerf.perturb"] if train else 0,
                N_importance=self.hparams["nerf.N_importance"],
                white_back=self.train_dataset.white_back,
                encode_feat=True if self.hparams["nerf.feat_dim"] > 0 else False,
                validation=False if train else True,
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def training_step(self, batch, batch_nb):
        ray_infos = batch["ray_infos"]
        rgbs = batch["rgbs"]
        img_idx = batch["img_idx"]
        directions = batch["directions"]
        pose = batch["c2w"]

        if self.hparams["pose_optimize"]:
            pose_refine = camera_utils.lie.se3_to_SE3(self.se3_refine(img_idx))
            refined_pose = camera_utils.pose.compose([pose_refine, pose])
            rays_o, rays_d = get_rays(directions, refined_pose)  # both (h*w, 3)
        else:
            rays_o, rays_d = get_rays(directions, pose)  # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d, ray_infos], 1)

        results = self(rays, img_idx)
        loss = ((results["s_rgb_fine"] - rgbs) ** 2).mean()

        if self.hparams["pose_optimize"]:
            self.optimizers()[0].zero_grad()
            self.optimizers()[1].zero_grad()
            self.manual_backward(loss)
            self.optimizers()[0].step()
            self.optimizers()[1].step()
        else:
            self.optimizers().zero_grad()
            self.manual_backward(loss)
            self.optimizers().step()

        with torch.no_grad():
            psnr_ = metric_utils.psnr(results[f"s_rgb_fine"], rgbs)
        self.log("lr", get_learning_rate(self.optimizer))
        if self.hparams["pose_optimize"]:
            self.log("lr_pose", get_learning_rate(self.optimizers()[1]))
        self.log("train/loss", loss)
        self.log("train/psnr", psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        ray_infos = batch["ray_infos"][0]
        rgbs = batch["rgbs"][0]
        img_idx = batch["img_idx"][0]
        directions = batch["directions"][0]
        pose = batch["c2w"][0]

        # get refined pose
        if self.hparams["pose_optimize"]:
            pose_refine = camera_utils.lie.se3_to_SE3(self.se3_refine(img_idx))
            refined_pose = camera_utils.pose.compose([pose_refine, pose])
            rays_o, rays_d = get_rays(directions, refined_pose)  # both (h*w, 3)
        else:
            rays_o, rays_d = get_rays(directions, pose.squeeze())  # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d, ray_infos], 1)

        # forward
        results = self(rays, img_idx, train=False)
        loss = ((results["s_rgb_fine"] - rgbs) ** 2).mean()

        # log
        idx = self.val_dataset.optimize_num
        if self.hparams["dataset_name"] in ["phototourism", "custom"]:
            WH = batch["img_wh"]
            W, H = WH[0].item(), WH[1].item()
        else:
            raise NotImplementedError
        img = results[f"s_rgb_fine"].view(H, W, -1).permute(2, 0, 1).cpu()  # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)

        psnr_ = metric_utils.psnr(results[f"s_rgb_fine"], rgbs)
        ssim_ = metric_utils.ssim(img[None, ...], img_gt[None, ...])
        lpips_ = metric_utils.lpips_alex((img_gt[None, ...]), img[None, ...])
        self.log("val/loss", loss)
        self.log("val/psnr", psnr_)
        self.log("val/ssim", ssim_)
        self.log("val/lpips", lpips_)

        ### log image
        if self.logger is not None:
            self.logger.log_image(f"val_{idx}/viz/GT", [img_gt])
            self.logger.log_image(f"val_{idx}/viz/rgb_fine", [img])

        if psnr_ > self.best_psnr:
            self.best_psnr = psnr_
            self.best_ssim = ssim_
            self.best_lpips = lpips_
            if self.hparams["pose_optimize"]:
                save_path = os.path.join(
                    self.save_dir,
                    "best_pose_" + str(self.hparams["optimize_num"]).zfill(2) + ".npy",
                )
                np.save(save_path, np.array(refined_pose[0].cpu()))
            else:
                save_path = os.path.join(
                    self.save_dir,
                    "best_pose_" + str(self.hparams["optimize_num"]).zfill(2) + ".npy",
                )
                np.save(save_path, np.array(self.embedding_fine_a(img_idx[0]).cpu()))
                if os.path.isfile(self.psnr_path):
                    with open(self.psnr_path, "rb") as f:
                        self.best_psnr_dict = pickle.load(f)
                    with open(self.ssim_path, "rb") as f:
                        self.best_ssim_dict = pickle.load(f)
                    with open(self.lpips_path, "rb") as f:
                        self.best_lpips_dict = pickle.load(f)
                self.best_psnr_dict[self.hparams["optimize_num"]] = self.best_psnr.cpu()
                self.best_ssim_dict[self.hparams["optimize_num"]] = self.best_ssim.cpu()
                self.best_lpips_dict[
                    self.hparams["optimize_num"]
                ] = self.best_lpips.cpu()
                with open(self.psnr_path, "wb") as f:
                    pickle.dump(self.best_psnr_dict, f)
                with open(self.ssim_path, "wb") as f:
                    pickle.dump(self.best_ssim_dict, f)
                with open(self.lpips_path, "wb") as f:
                    pickle.dump(self.best_lpips_dict, f)

    def validation_epoch_end(self, ourputs):
        pass

    def dataset_setup(self):
        dataset = dataset_dict[self.hparams["dataset_name"] + "_optimize"]
        kwargs = {"root_dir": self.hparams["root_dir"]}
        if self.hparams["dataset_name"] in ["phototourism", "custom"]:
            kwargs["scene_name"] = self.hparams["scene_name"]
            kwargs["img_downscale"] = self.hparams["phototourism.img_downscale"]
            kwargs["use_cache"] = self.hparams["phototourism.use_cache"]
            kwargs["near"] = self.hparams["nerf.near"]
            kwargs["far"] = self.hparams["nerf.far"]
            kwargs["pose_optimize"] = self.hparams["pose_optimize"]
            kwargs["optimize_num"] = self.hparams["optimize_num"]
        else:
            raise NotImplementedError
        self.train_dataset = dataset(
            split="train", camera_noise=self.hparams["pose.noise"], **kwargs
        )
        self.val_dataset = dataset(
            split="val", camera_noise=self.hparams["pose.noise"], **kwargs
        )

    @torch.no_grad()
    def model_setup(self):
        super().model_setup()
        N_images = self.train_dataset.N_images_test
        checkpoint = torch.load(self.hparams["ckpt_path"])
        self.embedding_fine_a = torch.nn.Embedding(
            N_images, self.hparams["nerf.appearance_dim"]
        )
        self.embeddings["fine_a"] = self.embedding_fine_a
        self.models_to_train = [self.embedding_fine_a]
        load_ckpt(self.nerf_coarse, self.hparams["ckpt_path"], model_name="nerf_coarse")
        load_ckpt(self.nerf_fine, self.hparams["ckpt_path"], model_name="nerf_fine")
        self.nerf_coarse.encode_candidate = False
        self.nerf_fine.encode_candidate = False
        # Approximate pose initialization using GT pose before pose optimizing.
        if self.hparams["pose_optimize"]:
            train_se3_refine = torch.nn.Embedding(
                self.train_dataset.N_images_train, 6
            ).to("cuda")
            train_se3_refine.weight[:] = checkpoint["state_dict"]["se3_refine.weight"]

            gt_train_poses = [
                v
                for k, v in self.train_dataset.GT_poses_dict.items()
                if k in self.train_dataset.img_ids_train
            ]
            gt_train_poses = torch.tensor(np.stack(gt_train_poses, 0))
            noise_poses = torch.stack([torch.eye(3, 4)] * len(gt_train_poses))
            pose_refine_ = camera_utils.lie.se3_to_SE3(train_se3_refine.weight).cpu()
            refine_poses = camera_utils.pose.compose([pose_refine_, noise_poses])

            refine_poses = torch.stack(
                [metric_utils.parse_raw_camera(p) for p in refine_poses.float()], dim=0
            )
            gt_train_poses = torch.stack(
                [metric_utils.parse_raw_camera(p) for p in gt_train_poses.float()],
                dim=0,
            )
            aligned_pose, sim3 = metric_utils.prealign_cameras(
                refine_poses, gt_train_poses
            )

            gt_test_poses = [
                v
                for k, v in self.train_dataset.GT_poses_dict.items()
                if k in self.train_dataset.img_ids_test
            ]
            gt_test_poses = torch.tensor(np.stack(gt_test_poses, 0))
            gt_test_poses_ = torch.stack(
                [metric_utils.parse_raw_camera(p) for p in gt_test_poses.float()], dim=0
            )
            center = torch.zeros(1, 1, 3)
            center_GT = camera_utils.cam2world(center, gt_test_poses_)[:, 0]  # [N,3]
            center_aligned = (
                center_GT - sim3.t0
            ) / sim3.s0 @ sim3.R * sim3.s1 + sim3.t1
            R_aligned = gt_test_poses_[..., :3] @ sim3.R
            t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
            aligned_GT_pose_ = camera_utils.pose(R=R_aligned, t=t_aligned)
            aligned_GT_pose = torch.stack(
                [metric_utils.parse_raw_camera(p) for p in aligned_GT_pose_.float()],
                dim=0,
            )

            for i, (k, v) in enumerate(self.train_dataset.poses_dict.items()):
                self.train_dataset.poses_dict[k] = aligned_GT_pose[i]
                self.val_dataset.poses_dict[k] = aligned_GT_pose[i]
        else:
            optimized_pose_path = os.path.join(
                self.pose_save_dir,
                "best_pose_" + str(self.hparams["optimize_num"]).zfill(2) + ".npy",
            )

            id_ = self.train_dataset.img_ids_test[self.hparams["optimize_num"]]
            self.train_dataset.poses_dict[id_] = torch.from_numpy(
                np.load(optimized_pose_path)
            )
            self.val_dataset.poses_dict[id_] = torch.from_numpy(
                np.load(optimized_pose_path)
            )
