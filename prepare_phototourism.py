import argparse
import os
import pickle

import numpy as np

from configs.config import parse_args
from datasets import PhototourismDataset, CustomDataset
from datasets import dataset_dict
from contextlib import contextmanager

@contextmanager
def open_file(name):
    f = open(name, 'wb')
    yield f
    print(f"Caching {name} complete")
    f.close()

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Path to config file.",
        required=False,
        default="./configs/phototourism.yaml",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Modify hparams. Example: train.py resume out_dir TRAIN.BATCH_SIZE 2",
    )
    return parse_args(parser)


if __name__ == "__main__":
    hparams = get_opts()
    root_dir = hparams["root_dir"]
    scale = hparams["phototourism.img_downscale"]
    cache_dir = os.path.join(root_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Preparing cache for scale {scale}. It can take several minutes...")
    DATASETCLS = dataset_dict[hparams["dataset_name"]]
    dataset = DATASETCLS(
        root_dir=root_dir,
        scene_name=hparams["scene_name"],
        feat_dir=hparams["feat_dir"],
        depth_dir=hparams["depth_dir"],
        split="train",
        img_downscale=scale,
        camera_noise=None,
    )
    # save img ids
    with open_file(os.path.join(cache_dir, f"img_ids.pkl")) as f:
        pickle.dump(dataset.img_ids, f, pickle.HIGHEST_PROTOCOL)

    # save img paths
    with open_file(os.path.join(cache_dir, f"image_paths.pkl")) as f:
        pickle.dump(dataset.image_paths, f, pickle.HIGHEST_PROTOCOL)

    # save Ks
    with open_file(os.path.join(cache_dir, f"Ks{scale}.pkl")) as f:
        pickle.dump(dataset.Ks, f, pickle.HIGHEST_PROTOCOL)

    # save scene points
    np.save(os.path.join(cache_dir, "xyz_world.npy"), dataset.xyz_world)

    # save poses
    np.save(os.path.join(cache_dir, "poses.npy"), dataset.poses)

    # save near and far bounds
    with open_file(os.path.join(cache_dir, f"nears.pkl")) as f:
        pickle.dump(dataset.nears, f, pickle.HIGHEST_PROTOCOL)
    with open_file(os.path.join(cache_dir, f"fars.pkl")) as f:
        pickle.dump(dataset.fars, f, pickle.HIGHEST_PROTOCOL)

    # save rays and rgbs
    with open_file(os.path.join(cache_dir, f"ray_infos{scale}.pkl")) as f:
        pickle.dump(dataset.all_ray_infos, f, pickle.HIGHEST_PROTOCOL)
    with open_file(os.path.join(cache_dir, f"rgbs{scale}.pkl")) as f:
        pickle.dump(dataset.all_rgbs, f, pickle.HIGHEST_PROTOCOL)
    with open_file(os.path.join(cache_dir, f"directions{scale}.pkl")) as f:
        pickle.dump(dataset.all_directions, f, pickle.HIGHEST_PROTOCOL)

    # save imgs_wh
    with open_file(os.path.join(cache_dir, f"all_imgs_wh{scale}.pkl")) as f:
        pickle.dump(dataset.all_imgs_wh, f, pickle.HIGHEST_PROTOCOL)

    # save feature maps
    with open_file(os.path.join(cache_dir, f"feat_maps{scale}.pkl")) as f:
        pickle.dump(dataset.feat_maps, f, pickle.HIGHEST_PROTOCOL)

    # save pxl coords
    with open_file(os.path.join(cache_dir, f"all_pxl_coords{scale}.pkl")) as f:
        pickle.dump(dataset.all_pxl_coords, f, pickle.HIGHEST_PROTOCOL)

    print(f"Data cache saved to {cache_dir}!")
