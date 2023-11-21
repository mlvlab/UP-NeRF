import lpips
import torch
from kornia.losses import ssim as dssim

import utils.camera as camera

lpips_alex = lpips.LPIPS(net="alex")


def mse(image_pred, image_gt, valid_mask=None, reduction="mean"):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == "mean":
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction="mean"):
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


def ssim(image_pred, image_gt, reduction="mean"):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim.ssim_loss(
        image_pred, image_gt, 3, reduction=reduction
    )  # dissimilarity in [0, 1]
    return 1 - 2 * dssim_  # in [-1, 1]


# camera pose
def parse_raw_camera(pose_raw):
    pose_flip = camera.pose(R=torch.diag(torch.tensor([1, -1, -1])))
    pose = camera.pose.compose([pose_flip, pose_raw[:3]])
    pose = camera.pose.invert(pose)
    pose = camera.pose.compose([pose_flip, pose])
    return pose


def prealign_cameras(pose, pose_GT):
    pose, pose_GT = pose.float(), pose_GT.float()
    center = torch.zeros(1, 1, 3)
    center_pred = camera.cam2world(center, pose)[:, 0]  # [N,3]
    center_GT = camera.cam2world(center, pose_GT)[:, 0]  # [N,3]
    sim3 = camera.procrustes_analysis(center_GT, center_pred)
    center_aligned = (center_pred - sim3.t1) / sim3.s1 @ sim3.R.t() * sim3.s0 + sim3.t0
    R_aligned = pose[..., :3] @ sim3.R.t()
    t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
    aligned_pose = camera.pose(R=R_aligned, t=t_aligned)
    return aligned_pose, sim3


def evaluate_camera_alignment(pose_aligned, pose_GT):
    # measure errors in rotation and translation
    R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)
    R_GT, t_GT = pose_GT.split([3, 1], dim=-1)
    R_error = camera.rotation_distance(R_aligned, R_GT)
    t_error = (t_aligned - t_GT)[..., 0].norm(dim=-1)
    error = dict(R=R_error, t=t_error)
    return error


def pose_metric(refine_poses, gt_poses):
    refine_poses = torch.stack(
        [parse_raw_camera(p) for p in refine_poses.float()], dim=0
    )
    gt_poses = torch.stack([parse_raw_camera(p) for p in gt_poses.float()], dim=0)
    try:
        aligned_pose, sim3 = prealign_cameras(refine_poses, gt_poses)
        error = evaluate_camera_alignment(aligned_pose, gt_poses)
    except:
        aligned_pose = refine_poses
        error = None
        print("pose alignment is not converged")
    return error, aligned_pose, gt_poses
