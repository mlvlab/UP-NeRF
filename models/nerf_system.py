import math
from collections import defaultdict

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.utils.data import DataLoader

import utils.camera as camera_utils
import utils.metric as metric_utils
import utils.pose_visualization as pose_viz_utils
import utils.ray as ray_utils
import utils.visualization as viz_utils
from datasets import dataset_dict
from losses import UPNeRFLoss
from models.nerf import NeRF
from models.rendering import render_rays
from models.transient_net import TransientNet
from utils.optim import get_learning_rate, get_optimizer, get_scheduler


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False
        self.candidate_schedule = hparams["candidate_schedule"]
        self.fine = True if hparams["nerf.N_importance"] > 0 else False
        self.loss = UPNeRFLoss(
            depth_mult=self.hparams["loss.depth_mult"],
            alpha_reg=self.hparams["loss.alpha_reg"],
            encode_feat=True if self.hparams["nerf.feat_dim"] > 0 else False,
            fine=self.fine,
        )
        self.val_log_N = 0

    def setup(self, stage):
        self.dataset_setup()
        self.model_setup()

    def configure_optimizers(self):
        self.optimizer = get_optimizer(
            self.hparams["optimizer.type"],
            self.hparams["optimizer.lr"],
            self.models_to_train,
        )
        scheduler_nerf = get_scheduler(
            self.hparams["optimizer.scheduler.type"],
            self.hparams["optimizer.lr"],
            self.hparams["optimizer.scheduler.lr_end"],
            self.hparams["max_steps"],
            self.optimizer,
        )
        optimizer = [self.optimizer]
        scheduler = [{"scheduler": scheduler_nerf, "interval": "step"}]

        if self.hparams["pose.optimize"]:
            self.optimizer_pose = get_optimizer(
                self.hparams["optimizer_pose.type"],
                self.hparams["optimizer_pose.lr"],
                [self.depth_scale, self.se3_refine],
            )
            scheduler_pose = get_scheduler(
                self.hparams["optimizer_pose.scheduler.type"],
                self.hparams["optimizer_pose.lr"],
                self.hparams["optimizer_pose.scheduler.lr_end"],
                self.hparams["max_steps"],
                self.optimizer_pose,
            )
            optimizer += [self.optimizer_pose]
            scheduler += [{"scheduler": scheduler_pose, "interval": "step"}]

        return optimizer, scheduler

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams["train.batch_size"],
            num_workers=self.hparams["train.num_workers"],
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,  # validate one image (H*W rays) at a time
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )

    def forward(self, rays, feats, img_idx, sched_mult, train=True):
        """Do batched inference on rays using chunk."""
        if sched_mult == 0:
            sched_phase = 0  # candidate only; feature only
        elif sched_mult == 1:
            sched_phase = 2  # no candidate; rgb only
        else:
            sched_phase = 1

        B = rays.shape[0]
        results = defaultdict(list)
        chunk = B if train else self.hparams["val.chunk_size"]
        for i in range(0, B, chunk):
            rendered_ray_chunks = render_rays(
                models=self.models,
                embeddings=self.embeddings,
                rays=rays[i : i + chunk],
                img_idx=img_idx[i : i + chunk],
                sched_mult=sched_mult,
                sched_phase=sched_phase,
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

        if sched_mult > 0:
            if self.transient_net is not None:
                t_results = self.transient_net(feats, img_idx)
                t_rgbs, t_alphas, t_betas = (
                    t_results["rgb"],
                    t_results["alpha"],
                    t_results["beta"],
                )
                results["rgb_coarse"] = (
                    results["s_rgb_coarse"] * (1 - t_alphas.detach())
                    + t_rgbs.detach() * t_alphas.detach()
                )
                results["rgb_fine"] = (
                    results["s_rgb_fine"] * (1 - t_alphas) + t_rgbs * t_alphas
                )
                results["t_beta"] = t_betas
                results["t_alpha"] = t_alphas
            else:
                results["rgb_coarse"] = results["s_rgb_coarse"]

        return results

    def training_step(self, batch, batch_nb):
        ray_infos = batch["ray_infos"]
        rgbs = batch["rgbs"]
        directions = batch["directions"]
        pose = batch["c2w"]
        feats = batch["feats"]
        img_idx = batch["img_idx"]

        if self.hparams["pose.optimize"]:
            pose_refine = camera_utils.lie.se3_to_SE3(self.se3_refine(img_idx))
            refined_pose = camera_utils.pose.compose([pose_refine, pose])
            rays_o, rays_d = ray_utils.get_rays(
                directions, refined_pose
            )  # both (h*w, 3)
        else:
            rays_o, rays_d = ray_utils.get_rays(directions, pose)  # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d, ray_infos], 1)

        # depth prediction
        inv_depths = batch["inv_depths"]
        scale, shift = torch.unbind(self.depth_scale(img_idx), 1)
        scale = torch.exp(scale)
        pred_inv_depths = inv_depths * scale + shift
        pred_inv_depths[pred_inv_depths < 1 / self.hparams["nerf.far"]] = (
            1 / self.hparams["nerf.far"]
        )
        pred_depths = 1.0 / pred_inv_depths
        pred_depths[pred_depths < self.hparams["nerf.near"]] = self.hparams["nerf.near"]

        # scheduling weight
        progress = self.nerf_coarse.progress.data.item()
        sched_mult = self.get_schedule_mult(progress)

        results = self(rays, feats, img_idx, sched_mult)

        loss_d = self.loss(results, rgbs, feats, pred_depths, sched_mult)
        loss = sum(l for l in loss_d.values())

        if self.hparams["pose.optimize"]:
            self.optimizers()[0].zero_grad()
            self.optimizers()[1].zero_grad()
            self.manual_backward(loss)
            self.optimizers()[0].step()
            self.lr_schedulers()[0].step()
            self.optimizers()[1].step()
            self.lr_schedulers()[1].step()
        else:
            self.optimizers().zero_grad()
            self.manual_backward(loss)
            self.optimizers().step()
            self.lr_schedulers().step()

        with torch.no_grad():
            typ = "fine" if self.fine else "coarse"
            if f"s_rgb_{typ}" in results:
                psnr_ = metric_utils.psnr(results[f"s_rgb_{typ}"], rgbs)
            else:
                psnr_ = torch.FloatTensor([0])

        if self.hparams["pose.optimize"]:
            self.log("lr", get_learning_rate(self.optimizers()[0].optimizer))
            self.log("lr_pose", get_learning_rate(self.optimizers()[1].optimizer))
        else:
            self.log("lr", get_learning_rate(self.optimizer))
        self.log("train/loss", loss)
        for k, v in loss_d.items():
            self.log(f"train/{k}", v, prog_bar=True)
        self.log("train/psnr", psnr_, prog_bar=True)

        if self.hparams["pose.optimize"]:
            if self.global_step % (self.hparams["train.log_pose_interval"] * 2) == 0:
                self.log_pose()
            self.nerf_coarse.progress.data.fill_(
                self.global_step / (self.hparams["max_steps"] * 2)
            )
            if self.fine:
                self.nerf_fine.progress.data.fill_(
                    self.global_step / (self.hparams["max_steps"] * 2)
                )
        return loss

    def validation_step(self, batch, batch_nb):
        ray_infos = batch["ray_infos"][0]  # (H*W, 2)
        rgbs = batch["rgbs"][0]  # (H*W, 3)
        img_idx = batch["img_idx"][0]  # (H*W,)
        directions = batch["directions"][0]  # (H*W, 3)
        pose = batch["c2w"][0]  # (3,4)
        feats = batch["feats"][0]  # (H*W, feat_dim)
        inv_depths = batch["inv_depths"][0]  # (H*W)

        if self.hparams["pose.optimize"]:
            pose_refine = camera_utils.lie.se3_to_SE3(self.se3_refine(img_idx))
            refined_pose = camera_utils.pose.compose([pose_refine, pose])
            rays_o, rays_d = ray_utils.get_rays(
                directions, refined_pose
            )  # both (H*W, 3)
        else:
            rays_o, rays_d = ray_utils.get_rays(directions, pose)  # both (H*W, 3)
        rays = torch.cat([rays_o, rays_d, ray_infos], 1)

        scale, shift = self.depth_scale(img_idx[0])
        scale = torch.exp(scale)
        pred_inv_depths = inv_depths * scale + shift
        pred_inv_depths[pred_inv_depths < 1 / self.hparams["nerf.far"]] = (
            1 / self.hparams["nerf.far"]
        )
        pred_depths = 1.0 / pred_inv_depths
        pred_depths[pred_depths < self.hparams["nerf.near"]] = self.hparams["nerf.near"]

        progress = self.nerf_coarse.progress.data.item()
        sched_mult = self.get_schedule_mult(progress)
        results = self(rays, feats, img_idx, sched_mult, train=False)

        loss_d = self.loss(results, rgbs, feats, pred_depths, sched_mult)
        loss = sum(l for l in loss_d.values())

        log = {"val_loss": loss}
        typ = "fine" if "rgb_fine" in results else "coarse"
        if f"rgb_{typ}" in results:
            psnr_ = metric_utils.psnr(results[f"rgb_{typ}"], rgbs)
            log["val_psnr"] = psnr_
        else:
            log["val_psnr"] = torch.FloatTensor([0])

        if self.hparams["debug"]:
            return log

        ### log image
        img_idx = img_idx[0].item()
        W, H = batch["img_wh"][0]
        pca_m = batch["pca_m"][0]  # (feat_dim,)
        pca_c = batch["pca_c"][0]  # (3, feat_dim)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)
        feat_gt = viz_utils.get_pca_img(feats.view(H, W, -1), pca_m, pca_c).permute(
            2, 0, 1
        )  # (3, H, W)
        min_ = results[f"s_depth_{typ}"].min().item()
        max_ = results[f"s_depth_{typ}"].max().item()
        rescale_depth_gt = viz_utils.visualize_depth(
            pred_depths.view(H, W), min_max=(min_, max_)
        )
        self.logger.log_image(f"val_{img_idx}/viz/rgb_GT", [img_gt])
        self.logger.log_image(f"val_{img_idx}/viz/feat_GT", [feat_gt])
        self.logger.log_image(f"val_{img_idx}/viz/rescale_depth_GT", [rescale_depth_gt])
        for log_img_name in self.hparams["val.log_image_list"]:
            try:
                if "depth" in log_img_name:
                    depth = results[log_img_name].view(H, W)  # (3, H, W)
                    img = viz_utils.visualize_depth(depth)
                elif "feat" in log_img_name and feats is not None:
                    feat_map = results[log_img_name].view(H, W, -1)
                    img = viz_utils.get_pca_img(feat_map, pca_m, pca_c).permute(2, 0, 1)
                elif "rgb" in log_img_name:
                    img = (
                        results[log_img_name].view(H, W, -1).permute(2, 0, 1).cpu()
                    )  # (3, H, W)
                self.logger.log_image(f"val_{img_idx}/viz/{log_img_name}", [img])
            except:
                pass

        return log

    def validation_epoch_end(self, outputs):
        if outputs == []:  # For pytorch lighting bug, when resumimg from checkpoint.
            return

        mean_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mean_psnr = torch.stack([x["val_psnr"] for x in outputs]).mean()
        self.log("val/loss", mean_loss)
        self.log("val/psnr", mean_psnr, prog_bar=True)

    def dataset_setup(self):
        dataset = dataset_dict[self.hparams["dataset_name"]]
        kwargs = {"root_dir": self.hparams["root_dir"]}
        if self.hparams["dataset_name"] == "phototourism":
            kwargs["scene_name"] = self.hparams["scene_name"]
            kwargs["img_downscale"] = self.hparams["phototourism.img_downscale"]
            kwargs["use_cache"] = self.hparams["phototourism.use_cache"]
            kwargs["feat_dir"] = self.hparams["feat_dir"]
            kwargs["depth_dir"] = self.hparams["depth_dir"]
            kwargs["near"] = self.hparams["nerf.near"]
            kwargs["far"] = self.hparams["nerf.far"]
        else:
            raise NotImplementedError
        self.train_dataset = dataset(
            split="train", camera_noise=self.hparams["pose.noise"], **kwargs
        )
        self.val_dataset = dataset(
            split="val",
            camera_noise=self.hparams["pose.noise"],
            val_img_idx=self.hparams["val.img_idx"],
            **kwargs,
        )

    def model_setup(self):
        N_images = self.train_dataset.N_images_train
        self.embeddings = {}
        self.models_to_train = []
        # Apperance embedding & Candidate embedding
        if self.hparams["nerf.appearance_dim"] > 0:
            self.embedding_coarse_a = torch.nn.Embedding(
                N_images, self.hparams["nerf.appearance_dim"]
            )
            self.embeddings["coarse_a"] = self.embedding_coarse_a
            self.models_to_train += [self.embedding_coarse_a]
            if self.fine:
                self.embedding_fine_a = torch.nn.Embedding(
                    N_images, self.hparams["nerf.appearance_dim"]
                )
                self.embeddings["fine_a"] = self.embedding_fine_a
                self.models_to_train += [self.embedding_fine_a]
        if self.hparams["nerf.candidate_dim"] > 0:
            self.embedding_coarse_c = torch.nn.Embedding(
                N_images, self.hparams["nerf.candidate_dim"]
            )
            self.embeddings["coarse_c"] = self.embedding_coarse_c
            self.models_to_train += [self.embedding_coarse_c]
            if self.fine:
                self.embedding_fine_c = torch.nn.Embedding(
                    N_images, self.hparams["nerf.candidate_dim"]
                )
                self.embeddings["fine_c"] = self.embedding_fine_c
                self.models_to_train += [self.embedding_fine_c]

        # NeRF model
        self.nerf_coarse = NeRF(
            "coarse",
            encode_feat=True if self.hparams["nerf.feat_dim"] > 0 else False,
            feat_dim=self.hparams["nerf.feat_dim"],
            xyz_L=self.hparams["nerf.N_emb_xyz"],
            dir_L=self.hparams["nerf.N_emb_dir"],
            appearance_dim=self.hparams["nerf.appearance_dim"],
            candidate_dim=self.hparams["nerf.candidate_dim"],
            c2f=self.hparams["pose.c2f"],
        )
        self.models = {"nerf_coarse": self.nerf_coarse}
        if self.fine:
            self.nerf_fine = NeRF(
                "fine",
                encode_feat=True if self.hparams["nerf.feat_dim"] > 0 else False,
                feat_dim=self.hparams["nerf.feat_dim"],
                xyz_L=self.hparams["nerf.N_emb_xyz"],
                dir_L=self.hparams["nerf.N_emb_dir"],
                appearance_dim=self.hparams["nerf.appearance_dim"],
                candidate_dim=self.hparams["nerf.candidate_dim"],
                c2f=self.hparams["pose.c2f"],
            )
            self.models["nerf_fine"] = self.nerf_fine

        # Transient network.
        self.transient_net = TransientNet(
            N_images=N_images,
            beta_min=self.hparams["t_net.beta_min"],
            trasient_dim=self.hparams["t_net.transient_dim"],
            feat_dim=self.hparams["t_net.feat_dim"],
        )
        self.models["transient_network"] = self.transient_net
        self.models_to_train += [self.models]

        # Pose estimation & Depth prior loss
        self.se3_refine = torch.nn.Embedding(N_images, 6).to("cuda")
        torch.nn.init.zeros_(self.se3_refine.weight)
        self.depth_scale = torch.nn.Embedding(N_images, 2).to("cuda")
        torch.nn.init.zeros_(self.depth_scale.weight)

    @torch.no_grad()
    def log_pose(self):
        if self.hparams["debug"]:
            return
        noised_poses = torch.stack(
            [self.train_dataset.poses_dict[i] for i in self.train_dataset.img_ids_train]
        )
        gt_poses = torch.stack(
            [
                torch.from_numpy(self.train_dataset.GT_poses_dict[i])
                for i in self.train_dataset.img_ids_train
            ]
        )
        pose_refine_ = camera_utils.lie.se3_to_SE3(self.se3_refine.weight).cpu()
        refine_poses = camera_utils.pose.compose([pose_refine_, noised_poses])

        pose_error, aligned_poses, gt_poses = metric_utils.pose_metric(
            refine_poses, gt_poses
        )
        if pose_error is not None:
            self.log("train/pose_R", pose_error["R"].mean() * 180 / torch.pi)
            self.log("train/pose_t", pose_error["t"].mean())

        if self.hparams["dataset_name"] == "phototourism":
            viz_N = 20
            pose_idx = list(range(viz_N))
            if pose_error is None:
                init_poses = aligned_poses
                img = pose_viz_utils.get_pose_image(
                    init_poses[pose_idx], gt_poses[pose_idx]
                )
                self.logger.log_image(f"train/init_pose", [img])
            else:
                img = pose_viz_utils.get_pose_image(
                    aligned_poses[pose_idx], gt_poses[pose_idx]
                )
                self.logger.log_image(f"train/refine_pose", [img])
        else:
            raise NotImplementedError

    def get_schedule_mult(self, progress):
        s, e = self.candidate_schedule
        if progress < s:
            sched_mult = 0
        elif progress > e:
            sched_mult = 1
        else:
            progress = (progress - s) / (e - s)
            sched_mult = (1 - math.cos(math.pi * progress)) / 2
        return sched_mult
