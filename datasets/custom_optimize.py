import os
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils.ray import *
import json

class CustomOptimizeDataset(Dataset):
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
        metadata = json.load(open(os.path.join(self.root_dir, "metadata.json"), 'r'))
        img_size = {}
        for id_, value in metadata.items():
            im = Image.open(os.path.join(self.root_dir, value['name']))
            img_size[id_] = im.size # (w, h)
        
        if self.use_cache:
            self.img_ids = load_pickle(os.path.join(self.cache_dir, "img_ids.pkl"))
            self.image_paths = load_pickle(
                os.path.join(self.cache_dir, "image_paths.pkl")
            )
        else:
            img_path_to_id = {}
            self.img_ids = []
            self.image_paths = {}
            for id_, value in metadata.items():
                name = value['name']
                img_path_to_id[name] = id_
                self.image_paths[id_]= name
                self.img_ids += [id_]

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            self.Ks = load_pickle(os.path.join(self.cache_dir, f"Ks{self.scale}.pkl"))
        else:
            self.Ks = {}  # {id: K}
            for id_, value in metadata.items():
                K = np.zeros((3, 3), dtype=np.float32)
                width, height = img_size[id_]
                K[0, 0] = value['focal'] / self.scale # fx
                K[1, 1] = value['focal'] / self.scale  # fy
                K[0, 2] = (width / 2) / self.scale  # cx
                K[1, 2] = (height / 2) / self.scale  # cy
                K[2, 2] = 1
                self.Ks[id_] = K
        if self.use_cache:
            self.poses = np.load(os.path.join(self.root_dir, "cache/poses.npy"))
        else:
            # Pose must be right up back!
            try:
                poses = []
                for id_, value in metadata.items():
                    poses.append(value["c2w"])
                self.poses = np.stack(poses, 0)
            except KeyError:
                self.poses = np.array([])
        assert len(self.poses) > 0, "TTO only supports datasets with GT poses!"
        self.GT_poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}

        # dummy for compatibility with prepare_phototourism.py
        if self.use_cache:
            self.nears = load_pickle(os.path.join(self.root_dir, "cache/nears.pkl"))
            self.fars = load_pickle(os.path.join(self.root_dir, "cache/fars.pkl"))
            self.xyz_world = np.load(os.path.join(self.root_dir, "cache/xyz_world.npy"))
        else:
            self.nears = np.array([])
            self.fars = np.array([])
            self.xyz_world = np.array([])
        

        self.img_ids_train = []
        self.img_ids_test = []
        for id_, value in metadata.items():
            if value['split'] == 'train':
                self.img_ids_train += [id_]
            elif value['split'] == 'test':
                self.img_ids_test += [id_]
            else:
                raise NotImplementedError
        self.N_images_train = len(self.img_ids_train)
        self.N_images_test = len(self.img_ids_test)

        # pose estimation
        self.poses_dict = {id_: torch.eye(3, 4) for i, id_ in enumerate(self.img_ids_test)}

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

def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data
