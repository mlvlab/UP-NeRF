import os
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils.colmap import read_cameras_binary, read_images_binary, read_points3d_binary
from utils.ray import *


class PhototourismOptimizeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        scene_name,
        near=0.0,
        far=5.0,
        camera_noise=-1,
        split="train",
        img_downscale=1,
        use_cache=False,
        pose_optimize=True,
        optimize_num=None,
    ):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        self.root_dir = root_dir
        self.scene_name = scene_name
        self.near, self.far = near, far
        self.camera_noise = camera_noise
        self.split = split
        self.img_downscale = img_downscale
        if split == "val":  # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(2, self.img_downscale)
        self.use_cache = use_cache
        self.pose_optimize = pose_optimize
        self.optimize_num = optimize_num
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        tsv = os.path.join(self.root_dir, f"{self.scene_name}.tsv")
        self.files = pd.read_csv(tsv, sep="\t")
        self.files = self.files[~self.files["id"].isnull()]  # remove data without id
        self.files.reset_index(inplace=True, drop=True)

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.root_dir, f"cache/img_ids.pkl"), "rb") as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.root_dir, f"cache/image_paths.pkl"), "rb") as f:
                self.image_paths = pickle.load(f)
        else:
            imdata = read_images_binary(
                os.path.join(self.root_dir, "dense/sparse/images.bin")
            )
            img_path_to_id = {}
            for v in imdata.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.image_paths = {}  # {id: filename}
            for filename in list(self.files["filename"]):
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            with open(
                os.path.join(self.root_dir, f"cache/Ks{self.img_downscale}.pkl"), "rb"
            ) as f:
                self.Ks = pickle.load(f)
        else:
            self.Ks = {}  # {id: K}
            camdata = read_cameras_binary(
                os.path.join(self.root_dir, "dense/sparse/cameras.bin")
            )
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata[id_]
                img_w, img_h = int(cam.params[2] * 2), int(cam.params[3] * 2)
                img_w_ = img_w // self.img_downscale
                img_h_ = img_h // self.img_downscale
                K[0, 0] = cam.params[0] * img_w_ / img_w  # fx
                K[1, 1] = cam.params[1] * img_h_ / img_h  # fy
                K[0, 2] = cam.params[2] * img_w_ / img_w  # cx
                K[1, 2] = cam.params[3] * img_h_ / img_h  # cy
                K[2, 2] = 1
                self.Ks[id_] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.root_dir, "cache/poses.npy"))
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.0]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0)  # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3]  # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # Step 4: correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(self.root_dir, "cache/xyz_world.npy"))
            with open(os.path.join(self.root_dir, f"cache/nears.pkl"), "rb") as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.root_dir, f"cache/fars.pkl"), "rb") as f:
                self.fars = pickle.load(f)
        else:
            pts3d = read_points3d_binary(
                os.path.join(self.root_dir, "dense/sparse/points3D.bin")
            )
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate(
                [self.xyz_world, np.ones((len(self.xyz_world), 1))], -1
            )
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {}  # {id_: distance}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[
                    :, :3
                ]  # xyz in the ith cam coordinate
                xyz_cam_i = xyz_cam_i[
                    xyz_cam_i[:, 2] > 0
                ]  # filter out points that lie behind the cam
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max()
            scale_factor = max_far / 5  # so that the max far is scaled to 5
            self.poses[..., 3] /= scale_factor
            for k in self.nears:
                self.nears[k] /= scale_factor
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}

        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_train = [
            id_
            for i, id_ in enumerate(self.img_ids)
            if self.files.loc[i, "split"] == "train"
        ]
        self.img_ids_test = [
            id_
            for i, id_ in enumerate(self.img_ids)
            if self.files.loc[i, "split"] == "test"
        ]
        self.N_images_train = len(self.img_ids_train)
        self.N_images_test = len(self.img_ids_test)

        # pose estimation
        self.GT_poses_dict = self.poses_dict
        self.poses_dict = {
            id_: torch.eye(3, 4) for i, id_ in enumerate(self.img_ids_test)
        }

        if self.split == "train":  # create buffer of all rays and rgb data
            id_ = self.img_ids_test[self.optimize_num]
            img = Image.open(
                os.path.join(self.root_dir, "dense/images", self.image_paths[id_])
            ).convert("RGB")
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w // self.img_downscale
                img_h = img_h // self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            if not self.pose_optimize:
                img = img[:, :, : img_w // 2].clone()
            img = img.view(3, -1).permute(1, 0)  # (h*w//2, 3) RGB
            self.all_rgbs = img
            if self.pose_optimize:
                self.all_imgs_wh = [img_w, img_h]
                directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            else:
                self.all_imgs_wh = [img_w // 2, img_h]
                directions = get_ray_directions(img_h, img_w, self.Ks[id_])[
                    :, : img_w // 2
                ].clone()
            self.all_directions = directions.view(-1, 3)
            self.all_ray_infos = torch.cat(
                [
                    self.near * torch.ones_like(self.all_directions[:, :1]),
                    self.far * torch.ones_like(self.all_directions[:, :1]),
                    self.optimize_num * torch.ones(len(self.all_directions), 1),
                ],
                1,
            )  # (h*w, 8)

        elif self.split == "val":
            id_ = self.img_ids_test[self.optimize_num]
            img = Image.open(
                os.path.join(self.root_dir, "dense/images", self.image_paths[id_])
            ).convert("RGB")
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w // self.img_downscale
                img_h = img_h // self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            if not self.pose_optimize:
                img = img[:, :, img_w // 2 :].clone()
            img = img.view(3, -1).permute(1, 0)  # (h*w//2, 3) RGB
            self.all_rgbs = img
            if self.pose_optimize:
                self.all_imgs_wh = [img_w, img_h]
                directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            else:
                self.all_imgs_wh = [img_w - img_w // 2, img_h]
                directions = get_ray_directions(img_h, img_w, self.Ks[id_])[
                    :, img_w // 2 :
                ].clone()
            self.all_directions = directions.view(-1, 3)
            self.all_ray_infos = torch.cat(
                [
                    self.near * torch.ones_like(self.all_directions[:, :1]),
                    self.far * torch.ones_like(self.all_directions[:, :1]),
                    self.optimize_num * torch.ones(len(self.all_directions), 1),
                ],
                1,
            )  # (h*w, 8)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == "train":
            return len(self.all_ray_infos)
        if self.split == "val":
            return 1

    def __getitem__(self, idx):
        img_idx = self.all_ray_infos[idx, 2].long()
        if self.split == "train":  # use data in the buffers
            sample = {
                "ray_infos": self.all_ray_infos[idx, :2],
                "directions": self.all_directions[idx],
                "img_idx": img_idx,
                "c2w": torch.FloatTensor(self.poses_dict[self.img_ids_test[img_idx]]),
                "rgbs": self.all_rgbs[idx],
            }

        elif self.split == "val":
            img_idx = self.all_ray_infos[:, 2].long()
            id_ = self.img_ids_test[img_idx[0].item()]
            sample = {
                "ray_infos": self.all_ray_infos[:, :2],
                "directions": self.all_directions,
                "img_idx": img_idx,
                "c2w": torch.FloatTensor(self.poses_dict[id_]),
                "rgbs": self.all_rgbs,
                "img_wh": self.all_imgs_wh,
            }

        return sample
