import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image


def visualize_depth(depth, cmap=cv2.COLORMAP_JET, min_max=None):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    if min_max is None:
        mi = np.min(x)  # get minimum depth
        ma = np.max(x)
    else:
        mi, ma = min_max
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = np.clip(x, 0, 1)
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_


def get_pca_img(feat, m, c):
    pc = (feat - m[None, None, :]) @ c.T
    M, m = pc.max(), pc.min()
    pc = (pc - m) / (M - m)
    return pc
