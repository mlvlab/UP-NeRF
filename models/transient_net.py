import torch
import torch.nn as nn


class TransientNet(nn.Module):
    def __init__(self, N_images, beta_min=0.1, trasient_dim=128, feat_dim=384):
        super(TransientNet, self).__init__()
        self.beta_min = beta_min
        self.trasient_dim = trasient_dim
        self.embedding_t = nn.Embedding(N_images, trasient_dim)
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.final_encoder = nn.Linear(256, 256)
        self.t_encoder = nn.Sequential(nn.Linear(256 + trasient_dim, 128), nn.ReLU())
        self.alpha_layer = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.beta_layer = nn.Sequential(nn.Linear(128, 1), nn.Softplus())
        self.rgb_layer = nn.Sequential(nn.Linear(128, 3), nn.Sigmoid())

    def forward(self, feat, ts):
        ret = {}
        t_emb = self.embedding_t(ts)  # (B, trasient_dim)
        feat_encoding = self.feat_encoder(feat)
        final_encoding = self.final_encoder(feat_encoding)
        temb_input = self.t_encoder(torch.cat([final_encoding, t_emb], -1))
        ret["alpha"] = self.alpha_layer(feat_encoding)  # (B, 1)
        ret["rgb"] = self.rgb_layer(temb_input)  # (B, 1)
        ret["beta"] = (
            self.beta_layer(temb_input) * ret["alpha"] + self.beta_min
        )  # (B, 1)
        return ret
