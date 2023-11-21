import torch
from torch import nn


class NeRF(nn.Module):
    def __init__(
        self,
        typ,
        D=8,
        W=256,
        skips=[4],
        encode_feat=True,
        feat_dim=384,
        xyz_L=10,
        dir_L=8,
        appearance_dim=48,
        candidate_dim=16,
        c2f=None,
    ):
        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.xyz_L = xyz_L
        self.dir_L = dir_L
        self.in_channels_xyz = 6 * xyz_L + 3
        self.in_channels_dir = 6 * dir_L + 3
        self.feat_dim = feat_dim
        self.appearance_dim = appearance_dim
        self.candidate_dim = candidate_dim
        self.encode_feat = encode_feat
        self.encode_appearance = True if appearance_dim > 0 else False
        self.encode_candidate = True if candidate_dim > 0 else False
        self.c2f = c2f
        self.progress = torch.nn.Parameter(torch.tensor(0.0))

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + self.in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # share head
        self.share_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        if self.encode_feat:
            self.feat_share_layer = nn.Linear(W, self.feat_dim)
            in_channels = self.feat_dim + self.in_channels_dir
        else:
            in_channels = W + self.in_channels_dir
        if self.encode_appearance:
            in_channels = in_channels + self.appearance_dim
        self.rgb_share_layer = nn.Sequential(
            nn.Linear(in_channels, W // 2),
            nn.ReLU(True),
            nn.Linear(W // 2, 3),
            nn.Sigmoid(),
        )

        # candidate head
        if self.encode_candidate:
            self.candidate_encoding = nn.Sequential(
                nn.Linear(W + candidate_dim, W // 2),
                nn.ReLU(True),
                nn.Linear(W // 2, W // 2),
                nn.ReLU(True),
            )
            self.candidate_sigma = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())
            if self.encode_feat:
                self.feat_candidate_layer = nn.Linear(W // 2, self.feat_dim)
            else:
                self.rgb_candidate_layer = nn.Linear(W // 2, 3)

    def forward(self, inputs, sched_mult, sigma_only=False):
        ret = {}
        input_xyz = self.positional_encoding(inputs["input_xyz"], self.xyz_L)
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        ret["s_sigma"] = self.share_sigma(xyz_)  # (B, 1)
        if sigma_only:
            return ret

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        if self.encode_feat:
            ret["s_feat"] = self.feat_share_layer(xyz_encoding_final)
            if sched_mult < 1 and self.encode_candidate:  # The candidate head affected.
                c_input = torch.cat([xyz_encoding_final, inputs["input_c"]], 1)
                c_input = self.candidate_encoding(c_input)
                ret["c_sigma"] = self.candidate_sigma(c_input)  # (B, 1)
                ret["c_feat"] = self.feat_candidate_layer(c_input)
            if sched_mult > 0:  #
                input_dir = self.positional_encoding(inputs["input_dir"], self.dir_L)
                if self.encode_appearance:
                    s_input = torch.cat(
                        [ret["s_feat"], input_dir, inputs["input_a"]], 1
                    )
                else:
                    s_input = torch.cat([ret["s_feat"], input_dir], 1)
                ret["s_rgb"] = self.rgb_share_layer(s_input)
        else:
            input_dir = self.positional_encoding(inputs["input_dir"], self.dir_L)
            if self.encode_appearance:
                s_input = torch.cat(
                    [xyz_encoding_final, input_dir, inputs["input_a"]], 1
                )
            else:
                s_input = torch.cat([xyz_encoding_final, input_dir], 1)
            ret["s_rgb"] = self.rgb_share_layer(s_input)
            if sched_mult < 1:  # The candidate head affected.
                c_input = torch.cat([xyz_encoding_final, inputs["input_c"]], 1)
                c_input = self.candidate_encoding(c_input)
                ret["c_sigma"] = self.candidate_sigma(c_input)  # (B, 1)
                ret["c_rgb"] = self.rgb_candidate_layer(c_input)
        return ret

    def positional_encoding(self, input, L):  # [B,...,N]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        shape = input.shape
        freq = (
            2 ** torch.arange(L, dtype=torch.float32, device=input.device) * torch.pi
        )  # [L]
        spectrum = input[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]

        if self.c2f is not None:
            # set weights for different frequency bands
            start, end = self.c2f
            alpha = (self.progress.data - start) / (end - start) * L
            k = torch.arange(L, dtype=torch.float32, device=input_enc.device)
            weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(torch.pi).cos_()) / 2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1, L) * weight).view(*shape)
        input_enc = torch.cat([input, input_enc], -1)
        return input_enc
