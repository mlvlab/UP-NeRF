import os
import pickle

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# barf
import utils.camera as camera_utils
from utils.colmap import read_cameras_binary, read_images_binary, read_points3d_binary
from utils.ray import *


class PhototourismDataset(Dataset):
    def __init__(
        self,
        root_dir,
        scene_name,
        feat_dir=None,
        depth_dir=None,
        near=0.1,
        far=5.0,
        camera_noise=-1,
        split="train",
        img_downscale=1,
        val_img_idx=[0],
        use_cache=False,
    ):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        self.root_dir = root_dir        
        self.scene_name = scene_name
        if feat_dir is not None:
            self.feat_map_dir = os.path.join(feat_dir, "feature_maps")
            self.pca_info_dir = os.path.join(feat_dir, "pca_infos")
        self.depth_dir = depth_dir
        self.near, self.far = near, far
        self.camera_noise = camera_noise
        self.split = split
        self.scale = img_downscale
        if split == "val":  # image downscale=1 will cause OOM in val mode
            self.scale = max(2, self.scale)
        self.val_img_idx = val_img_idx
        self.use_cache = use_cache
        self.cache_dir = os.path.join(root_dir, "cache") if self.use_cache else None
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
            self.img_ids = load_pickle(os.path.join(self.cache_dir, "img_ids.pkl"))
            self.image_paths = load_pickle(
                os.path.join(self.cache_dir, "image_paths.pkl")
            )
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
            self.Ks = load_pickle(os.path.join(self.cache_dir, f"Ks{self.scale}.pkl"))
        else:
            self.Ks = {}  # {id: K}
            camdata = read_cameras_binary(
                os.path.join(self.root_dir, "dense/sparse/cameras.bin")
            )
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata[id_]
                img_w, img_h = int(cam.params[2] * 2), int(cam.params[3] * 2)
                img_w_ = img_w // self.scale
                img_h_ = img_h // self.scale
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
            self.nears = load_pickle(os.path.join(self.cache_dir, "nears.pkl"))
            self.fars = load_pickle(os.path.join(self.cache_dir, "fars.pkl"))
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
        poses = []
        self.id2idx = {}
        for idx, id_ in enumerate(self.img_ids_train):
            self.id2idx[id_] = idx
            poses += [self.poses_dict[id_]]
        poses = np.stack(poses, 0)
        poses = torch.FloatTensor(poses)

        if self.camera_noise is not None:
            self.GT_poses_dict = self.poses_dict
            noise_file_name = f"./noises/{len(poses)}_{str(self.camera_noise)}.pt"
            if os.path.isfile(noise_file_name):
                self.pose_noises = torch.load(noise_file_name)
                poses = camera_utils.pose.compose([self.pose_noises, poses])
                self.poses_dict = {
                    id_: poses[i] for i, id_ in enumerate(self.img_ids_train)
                }
                print("load noise file: ", noise_file_name)
            else:
                if self.camera_noise == -1:
                    # intialize poses at one view point.
                    self.poses_dict = {
                        id_: torch.eye(3, 4) for i, id_ in enumerate(self.img_ids_train)
                    }
                else:
                    se3_noise = torch.randn(len(poses), 6) * self.camera_noise
                    self.pose_noises = camera_utils.lie.se3_to_SE3(se3_noise)
                    os.makedirs("./nosies", exist_ok=True)
                    torch.save(self.pose_noises, noise_file_name)
                    poses = camera_utils.pose.compose([self.pose_noises, poses])
                    self.poses_dict = {
                        id_: poses[i] for i, id_ in enumerate(self.img_ids_train)
                    }

        if self.split == "train":  # create buffer of all rays and rgb data
            if self.use_cache:
                self.all_ray_infos = load_pickle(
                    os.path.join(self.cache_dir, f"ray_infos{self.scale}.pkl")
                )  # (N_rays, 3)
                self.all_rgbs = load_pickle(
                    os.path.join(self.cache_dir, f"rgbs{self.scale}.pkl")
                )  # (N_rays, 3)
                self.all_directions = load_pickle(
                    os.path.join(self.cache_dir, f"directions{self.scale}.pkl")
                )  # (N_rays, 3)
                self.all_imgs_wh = load_pickle(
                    os.path.join(self.cache_dir, f"all_imgs_wh{self.scale}.pkl")
                )  # (N_rays, 2)
                self.feat_maps = load_pickle(
                    os.path.join(self.cache_dir, f"feat_maps{self.scale}.pkl")
                )  # (N_imgs, feat_H, feat_W, feat_dim)
                self.all_pxl_coords = load_pickle(
                    os.path.join(self.cache_dir, f"all_pxl_coords{self.scale}.pkl")
                )  # (N_rays, 1) in [0, 1]

                if self.camera_noise is not None:  # use predefined near and far
                    self.all_ray_infos[:, 0] = self.near
                    self.all_ray_infos[:, 1] = self.far
            else:
                self.all_ray_infos = []
                self.all_rgbs = []
                self.all_directions = []
                self.all_imgs_wh = []
                for id_ in self.img_ids_train:
                    img = Image.open(
                        os.path.join(
                            self.root_dir, "dense/images", self.image_paths[id_]
                        )
                    ).convert("RGB")
                    img_w, img_h = img.size
                    if self.scale > 1:
                        img_w = img_w // self.scale
                        img_h = img_h // self.scale
                        img = img.resize((img_w, img_h), Image.LANCZOS)
                    img = self.transform(img)  # (3, img_h, img_w)
                    img = img.view(3, -1).permute(1, 0)  # (img_h*img_w, 3) RGB
                    self.all_rgbs += [img]
                    self.all_imgs_wh += [[img_w, img_h]]

                    directions = get_ray_directions(img_h, img_w, self.Ks[id_]).view(
                        -1, 3
                    )
                    self.all_directions += [directions]

                    self.all_ray_infos += [
                        torch.cat(
                            [
                                self.nears[id_] * torch.ones_like(directions[:, :1]),
                                self.fars[id_] * torch.ones_like(directions[:, :1]),
                                self.id2idx[id_] * torch.ones(len(directions), 1),
                            ],
                            1,
                        )
                    ]  # (N_rays, 3)

                self.all_ray_infos = torch.cat(self.all_ray_infos, 0)  # (N_rays, 3)
                self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (N_rays, 3)
                self.all_directions = torch.cat(self.all_directions, 0)
                self.all_imgs_wh = torch.tensor(self.all_imgs_wh)

                self.feat_maps = []

                for id_, (W, H) in zip(self.img_ids_train, self.all_imgs_wh):
                    f_n = self.image_paths[id_].replace(".jpg", ".npy")
                    feat_map = np.load(
                        os.path.join(self.feat_map_dir, f_n)
                    )  # (feat_H, feat_W, feat_dim)
                    feat_map = torch.from_numpy(feat_map)
                    feat_map = feat_map / torch.norm(feat_map, dim=-1, keepdim=True)
                    self.feat_maps.append(feat_map)
                self.feat_maps = torch.stack(
                    self.feat_maps, 0
                )  # (N_imgs, feat_H, feat_W, feat_dim)

                # To get interpolation features efficiently
                self.all_pxl_coords = []
                for img_w, img_h in self.all_imgs_wh:
                    h_pxl = torch.linspace(0, img_h - 1, img_h) / (img_h - 1)
                    w_pxl = torch.linspace(0, img_w - 1, img_w) / (img_w - 1)
                    h_, w_ = torch.meshgrid(h_pxl, w_pxl, indexing="ij")
                    pxl = torch.stack((h_, w_), -1)
                    self.all_pxl_coords += [pxl.view(-1, 2)]
                self.all_pxl_coords = torch.cat(
                    self.all_pxl_coords, 0
                )  # (N_rays, 1) in [0, 1]

                if self.camera_noise is not None:  # use predefined near and far
                    self.all_ray_infos[:, 0] = self.near
                    self.all_ray_infos[:, 1] = self.far

            # Load inv_depth
            if self.depth_dir:
                self.all_inv_depths = []
                for id_, (W, H) in zip(self.img_ids_train, self.all_imgs_wh):
                    f_n = self.image_paths[id_].replace(".jpg", ".npy")
                    W, H = int(W), int(H)
                    inv_depth_map = np.load(os.path.join(self.depth_dir, f_n)).astype(
                        np.float32
                    )  # [C,H',W']
                    inv_depth_map[inv_depth_map < 0] = 0
                    M, m = 1 / self.near, 1 / self.far
                    inv_depth_map = inv_depth_map / inv_depth_map.max() * (M - m) + m
                    inv_depth_map = torch.from_numpy(cv2.resize(inv_depth_map, (W, H)))
                    self.all_inv_depths += [inv_depth_map.view(-1)]
                self.all_inv_depths = torch.cat(self.all_inv_depths, 0)  # (N_rays, 1)

        elif self.split in ["val"]:  # Use the train images as val images
            self.rgbs = []
            self.imgs_wh = []
            self.directions = []
            self.ray_infos = []
            self.feats = []
            self.pca_m = []
            self.pca_c = []
            self.inv_depths = []
            for img_idx in self.val_img_idx:
                id_ = self.img_ids_train[img_idx]
                img = Image.open(
                    os.path.join(self.root_dir, "dense/images", self.image_paths[id_])
                ).convert("RGB")
                img_w, img_h = img.size
                if self.scale > 1:
                    img_w = img_w // self.scale
                    img_h = img_h // self.scale
                    img = img.resize((img_w, img_h), Image.LANCZOS)
                img = self.transform(img)  # (3,img_h, img_w)
                self.rgbs += [img.view(3, -1).permute(1, 0)]  # (img_h*img_w, 3) RGB
                self.imgs_wh += [torch.LongTensor([img_w, img_h])]

                self.directions += [
                    get_ray_directions(img_h, img_w, self.Ks[id_]).view(-1, 3)
                ]
                if self.camera_noise != 0:
                    near = self.near
                    far = self.far
                else:
                    near = self.nears[id_]
                    far = self.fars[id_]
                ray_info = torch.cat(
                    [
                        near * torch.ones((img_w * img_h, 1)),
                        far * torch.ones((img_w * img_h, 1)),
                        self.id2idx[id_] * torch.ones((img_w * img_h, 1)),
                    ],
                    1,
                )  # (img_h*img_w, 3)
                self.ray_infos += [ray_info]

                f_n = self.image_paths[self.img_ids_train[img_idx]].replace(
                    ".jpg", ".npy"
                )
                feat_map = np.load(os.path.join(self.feat_map_dir, f_n))  # (H',W',C)
                feat_map = feat_map / np.linalg.norm(feat_map, axis=-1, keepdims=True)
                feat_map = torch.from_numpy(
                    cv2.resize(feat_map, (img_w, img_h))
                )  # (H,W,C)
                self.feats += [feat_map.view(img_w * img_h, -1)]
                self.pca_m += [
                    np.load(
                        os.path.join(
                            self.pca_info_dir, f_n.replace(".npy", "_mean.npy")
                        )
                    )
                ]
                self.pca_c += [
                    np.load(
                        os.path.join(
                            self.pca_info_dir, f_n.replace(".npy", "_components.npy")
                        )
                    )
                ]

                inv_depth_map = np.load(os.path.join(self.depth_dir, f_n)).astype(
                    np.float32
                )  # [C,H',W']
                inv_depth_map[inv_depth_map < 0] = 0
                M, m = 1 / near, 1 / far
                inv_depth_map = inv_depth_map / inv_depth_map.max() * (M - m) + m
                inv_depth_map = torch.from_numpy(
                    cv2.resize(inv_depth_map, (img_w, img_h))
                )
                self.inv_depths += [inv_depth_map.view(img_w * img_h)]

        else:  # TODO
            # for testing, create a parametric rendering path
            # test poses and appearance index are defined in eval.py
            self.poses_test = []
            for id in self.img_ids_test:
                self.poses_test += [self.poses_dict[id]]

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == "train":
            return len(self.all_ray_infos)
        if self.split == "val":
            return len(self.val_img_idx)
        if self.split == "test":  # TODO
            return self.N_images_test

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            img_idx = self.all_ray_infos[idx, 2].long()  # img id of ray
            sample = {
                "ray_infos": self.all_ray_infos[idx, :2],
                "directions": self.all_directions[idx],
                "img_idx": img_idx,
                "c2w": torch.FloatTensor(self.poses_dict[self.img_ids_train[img_idx]]),
                "rgbs": self.all_rgbs[idx],
            }
            if self.feat_map_dir is not None:
                h, w, c = self.feat_maps[img_idx].shape
                assert h == w
                points_mult = self.all_pxl_coords[idx] * (h - 1)
                y, x = points_mult
                y1, x1 = torch.floor(points_mult).long()
                y2, x2 = min(h - 1, y1 + 1), min(h - 1, x1 + 1)
                pixel11 = self.feat_maps[img_idx, y1, x1]
                pixel12 = self.feat_maps[img_idx, y1, x2]
                pixel21 = self.feat_maps[img_idx, y2, x1]
                pixel22 = self.feat_maps[img_idx, y2, x2]

                weight_11 = (y2 - y) * (x2 - x)
                weight_12 = (y2 - y) * (x - x1)
                weight_21 = (y - y1) * (x2 - x)
                weight_22 = (y - y1) * (x - x1)

                result = (
                    weight_11 * pixel11
                    + weight_12 * pixel12
                    + weight_21 * pixel21
                    + weight_22 * pixel22
                )
                sample["feats"] = result
                sample["inv_depths"] = self.all_inv_depths[idx]

        elif self.split == "val":
            sample = {}
            img_idx = self.ray_infos[idx][:, 2].long()
            id_ = self.img_ids_train[img_idx[0].item()]

            sample["rgbs"] = self.rgbs[idx]
            sample["ray_infos"] = self.ray_infos[idx][:, :2]
            sample["directions"] = self.directions[idx]
            sample["img_idx"] = img_idx
            sample["img_wh"] = self.imgs_wh[idx]
            sample["c2w"] = torch.FloatTensor(self.poses_dict[id_])
            sample["feats"] = self.feats[idx]
            sample["pca_m"] = self.pca_m[idx]
            sample["pca_c"] = self.pca_c[idx]
            sample["inv_depths"] = self.inv_depths[idx]

        elif self.split == "video":  # TODO
            sample = {}
            sample["c2w"] = c2w = torch.FloatTensor(self.poses_test[idx])
            directions = get_ray_directions(
                self.test_img_h, self.test_img_w, self.test_K
            )
            rays_o, rays_d = get_rays(directions, c2w)
            near, far = 0, 5
            rays = torch.cat(
                [
                    rays_o,
                    rays_d,
                    near * torch.ones_like(rays_o[:, :1]),
                    far * torch.ones_like(rays_o[:, :1]),
                ],
                1,
            )
            sample["rays"] = rays
            sample["ts"] = self.test_appearance_idx * torch.ones(
                len(rays), dtype=torch.long
            )
            sample["img_wh"] = torch.LongTensor([self.test_img_w, self.test_img_h])

        else:  # test #TODO
            sample = {}
            sample["c2w"] = c2w = torch.FloatTensor(self.poses_test[idx])
            id_ = self.img_ids_test[idx]
            img = Image.open(
                os.path.join(self.root_dir, "dense/images", self.image_paths[id_])
            ).convert("RGB")
            img_w, img_h = img.size
            if self.scale > 1:
                img_w = img_w // self.scale
                img_h = img_h // self.scale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            sample["rgbs"] = img

            directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat(
                [
                    rays_o,
                    rays_d,
                    self.nears[id_] * torch.ones_like(rays_o[:, :1]),
                    self.fars[id_] * torch.ones_like(rays_o[:, :1]),
                ],
                1,
            )  # (h*w, 8)
            sample["rays"] = rays
            sample["ts"] = self.test_appearance_idx * torch.ones(
                len(rays), dtype=torch.long
            )
            sample["img_wh"] = torch.LongTensor([img_w, img_h])

        return sample


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data
