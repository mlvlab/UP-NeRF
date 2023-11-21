import torch
from torch import nn


def L1_loss(pred, target):
    return torch.abs(pred - target)


def L2_loss(pred, target):
    return (pred - target) ** 2


class UPNeRFLoss(nn.Module):
    def __init__(self, depth_mult=1e-4, alpha_reg=1.0, encode_feat=True, fine=True):
        super().__init__()
        self.depth_mult = depth_mult
        self.alpha_reg = alpha_reg
        self.encode_feat = encode_feat
        self.fine = fine

    def forward(self, inputs, rgb_targets, feat_targets, depth_targets, schedule_mult):
        ret = {}
        # coarse network
        if schedule_mult < 1:  # sched_pahs <= 1
            l_depth_c = L1_loss(inputs["s_depth_coarse"], depth_targets)
            if "t_weight_coarse" in inputs.keys():
                l_depth_c *= 1 - inputs["t_weight_coarse"].detach()
            ret["l_depth_c"] = l_depth_c.mean() * self.depth_mult * (1 - schedule_mult)

            if self.encode_feat:
                l_feat_c = L2_loss(inputs["feat_coarse"], feat_targets)
                ret["l_feat_c"] = l_feat_c.mean() * (1 - schedule_mult)
            else:
                l_c_rgb_c = L2_loss(inputs["c_rgb_coarse"], rgb_targets)
                ret["l_c_rgb_c"] = l_c_rgb_c.mean() * (1 - schedule_mult)

        if schedule_mult > 0:  # sched_pahs >= 1
            l_rgb_c = L2_loss(inputs["s_rgb_coarse"], rgb_targets)
            ret["l_rgb_c"] = l_rgb_c.mean() * schedule_mult / 2

        if not self.fine:
            return ret

        # fine network
        if schedule_mult < 1:
            l_depth_f = L1_loss(inputs["s_depth_fine"], depth_targets)
            if "t_weight_fine" in inputs.keys():
                l_depth_f *= 1 - inputs["t_weight_fine"].detach()
            ret["l_depth_f"] = l_depth_f.mean() * self.depth_mult * (1 - schedule_mult)

            if self.encode_feat:
                l_feat_f = L2_loss(inputs["feat_fine"], feat_targets)
                ret["l_feat_f"] = l_feat_f.mean() * (1 - schedule_mult)
            else:
                l_c_rgb_f = L2_loss(inputs["c_rgb_fine"], rgb_targets)
                ret["l_c_rgb_f"] = l_c_rgb_f.mean() * (1 - schedule_mult)

        if schedule_mult > 0:
            l_rgb_f = L2_loss(inputs["s_rgb_fine"], rgb_targets)
            l_rgb_f = l_rgb_f / (2 * inputs["t_beta"] ** 2)
            ret["l_rgb_f"] = l_rgb_f.mean() * schedule_mult
            ret["l_beta"] = torch.log(inputs["t_beta"]).mean() * schedule_mult
            ret["l_alpha"] = inputs["t_alpha"].mean() * self.alpha_reg * schedule_mult
        return ret
