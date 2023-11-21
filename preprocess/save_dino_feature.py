import sys

sys.path.append("./dino-vit-features")

import argparse
import os

import numpy as np
import pandas as pd
import torch
from extractor import ViTExtractor
from sklearn.decomposition import PCA
from tqdm import tqdm


def main(args):
    extractor = ViTExtractor(device="cuda")
    extractor.model.eval()
    os.makedirs(args.save_dir, exist_ok=True)
    feat_save_dir = os.path.join(args.save_dir, "feature_maps")
    pca_save_dir = os.path.join(args.save_dir, "pca_infos")
    os.makedirs(feat_save_dir, exist_ok=True)
    os.makedirs(pca_save_dir, exist_ok=True)

    if args.tsv_path is None:
        file_names = os.path.listdir(args.image_dir)
    else:
        files = pd.read_csv(args.tsv_path, sep="\t")
        files = files[~files["id"].isnull()]  # remove data without id
        file_names = files["filename"]
    for f_n in tqdm(file_names):
        with torch.no_grad():
            # image_batch, image_pil = extractor.preprocess(os.path.join(args.image_dir, f_n), load_size=(args.resize, args.resize), scale_down=args.scale_down)
            image_batch, image_pil = extractor.preprocess(
                os.path.join(args.image_dir, f_n), load_size=(args.resize, args.resize)
            )
            descriptors = extractor.extract_descriptors(image_batch.cuda(), layer=9)
            H, W = extractor.num_patches
            feature = descriptors.squeeze().view(H, W, 384).cpu().numpy()
            np.save(os.path.join(feat_save_dir, f_n[:-4]), feature)

            feat_ = feature.reshape(-1, 384)
            feat_ = feat_ / np.linalg.norm(feat_, axis=-1, keepdims=True)
            pca = PCA(n_components=3)
            pca.fit(feat_)
            components = pca.components_
            mean = pca.mean_
            np.save(os.path.join(pca_save_dir, f_n[:-4] + "_mean.npy"), mean)
            np.save(
                os.path.join(pca_save_dir, f_n[:-4] + "_components.npy"), components
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="directory of data")
    parser.add_argument("--save_dir", required=True, help="save directory")
    parser.add_argument(
        "--tsv_path",
        default=None,
        help="Set tsv_path to save only the features that will be used for training.",
    )
    parser.add_argument("--scale_down", default=2, help="image scale down")
    parser.add_argument("--resize", default=448, help="image resize")
    args = parser.parse_args()

    main(args)
