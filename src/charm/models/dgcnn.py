"""DGCNN landmark network.

Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", ACM TOG 2019.
Adapted from PointNeXt (https://github.com/guochengqian/PointNeXt, MIT).
"""

from __future__ import annotations

import torch
from torch import nn

from charm.models.blocks import create_convblock1d, create_convblock2d
from charm.models.char import LandmarkNet, presence_head
from charm.models.ops import grouping_operation


@torch.no_grad()
def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """Indices (B, N, k) of the k nearest neighbours of x (B, N, C)."""
    dist = torch.cdist(x, x)
    return dist.topk(k=k, dim=-1, largest=False, sorted=True).indices


class EdgeConv(nn.Module):
    """Edge convolution: max over neighbours of h([x_i, x_j - x_i])."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.nn = create_convblock2d(in_channels * 2, out_channels, **kwargs)

    def forward(self, x, edge_index):
        x_j = grouping_operation(x.squeeze(-1), edge_index)
        features = torch.cat(
            [x.expand(-1, -1, -1, edge_index.shape[-1]), x_j - x], dim=1
        )
        return torch.max(self.nn(features), -1, keepdim=True)[0]


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.gconv = EdgeConv(in_channels, out_channels, **kwargs)

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv(GraphConv):
    """Edge convolution on the k-nearest-neighbour graph of the features."""

    def __init__(self, in_channels, out_channels, k=9, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.k = k

    def forward(self, x):
        edge_index = knn_graph(x.squeeze(-1).transpose(1, 2), self.k)
        return super().forward(x, edge_index)


class DGCNN(LandmarkNet):
    """DGCNN with a CHaR variant.

    The presence head reads the fused point features pooled with max and
    average.
    """

    def __init__(
        self,
        variant: str = "charm",
        num_landmarks: int = 80,
        num_teeth: int = 16,
        in_channels: int = 6,
        channels: int = 64,
        embed_dim: int = 1024,
        n_blocks: int = 5,
        k: int = 20,
        dropout: float = 0.3,
    ):
        super().__init__(variant, num_landmarks, num_teeth)
        norm_args = {"norm": "bn"}
        act_args = {"act": "leakyrelu", "negative_slope": 0.2}
        block_args = {"norm_args": norm_args, "act_args": act_args}
        self.n_blocks = n_blocks
        self.k = k
        self.head = GraphConv(in_channels, channels, **block_args)
        out_channels = [channels]
        in_channels = channels
        backbone = []
        for _ in range(n_blocks - 2):
            backbone.append(DynConv(in_channels, channels, k, **block_args))
            out_channels.append(channels)
            in_channels = channels
            channels *= 2
        self.backbone = nn.Sequential(*backbone)
        self.fusion_block = create_convblock1d(
            sum(out_channels), embed_dim, bias=False, **block_args
        )
        self.heat_regressor = nn.Sequential(
            nn.Conv1d(embed_dim, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_landmarks, 1),
            nn.Sigmoid(),
        )
        if self.has_presence_head:
            self.cls_head = presence_head(
                embed_dim * 2, (512, 256), num_teeth, dropout
            )

    def forward(self, points: torch.Tensor, normals: torch.Tensor):
        features = torch.cat([points, normals], dim=-1).transpose(1, 2)
        features = features.contiguous().unsqueeze(-1)
        feats = [self.head(features, knn_graph(points, self.k))]
        for block in self.backbone:
            feats.append(block(feats[-1]))
        fusion = self.fusion_block(torch.cat(feats, dim=1).squeeze(-1))
        heatmaps = self.heat_regressor(fusion)

        logits = None
        if self.has_presence_head:
            pooled = torch.cat(
                (fusion.max(dim=-1)[0], fusion.mean(dim=-1)), dim=1
            )
            logits = self.cls_head(pooled)
        return self.finish(heatmaps, logits)
