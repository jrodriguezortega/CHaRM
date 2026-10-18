"""PointMLP landmark network.

Ma et al., "Rethinking Network Design and Local Geometry in Point Cloud: A
Simple Residual MLP Framework", ICLR 2022. Adapted from
https://github.com/ma-xu/pointMLP-pytorch (Apache-2.0).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from charm.models.char import LandmarkNet, presence_head


def get_activation(activation: str) -> nn.Module:
    activation = activation.lower()
    if activation == "gelu":
        return nn.GELU()
    if activation == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    return nn.ReLU(inplace=True)


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Squared distances (B, N, M) between (B, N, C) and (B, M, C)."""
    B, N, _ = src.shape
    M = dst.shape[1]
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Select (B, S[, K], C) from points (B, N, C) by idx (B, S[, K])."""
    B = points.shape[0]
    view = [B] + [1] * (idx.dim() - 1)
    batch = torch.arange(B, device=points.device).view(view).expand_as(idx)
    return points[batch, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling from a random first point, (B, npoint)."""
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    batch_indices = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor):
    """Indices (B, S, nsample) of the nearest points of each query."""
    sqrdists = square_distance(new_xyz, xyz)
    return torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)[
        1
    ]


class LocalGrouper(nn.Module):
    """Sample anchors and group their k nearest neighbours.

    Takes xyz (B, N, 3) and features (B, N, D); returns the anchors
    (B, G, 3) and the grouped features (B, G, K, 2D [+ 3]).
    """

    def __init__(
        self,
        channel: int,
        groups: int,
        kneighbors: int,
        use_xyz: bool = True,
        normalize: str | None = "anchor",
    ):
        super().__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.normalize = normalize.lower() if normalize else None
        if self.normalize not in ("center", "anchor", None):
            raise ValueError("normalize must be center, anchor or None")
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(
                torch.ones([1, 1, 1, channel + add_channel])
            )
            self.affine_beta = nn.Parameter(
                torch.zeros([1, 1, 1, channel + add_channel])
            )

    def forward(self, xyz, points):
        B = xyz.shape[0]
        S = self.groups
        xyz = xyz.contiguous()
        fps_idx = farthest_point_sample(xyz, self.groups)
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            else:
                mean = (
                    torch.cat([new_points, new_xyz], dim=-1)
                    if self.use_xyz
                    else new_points
                )
                mean = mean.unsqueeze(dim=-2)
            std = torch.std(
                (grouped_points - mean).reshape(B, -1), dim=-1
            ).view(B, 1, 1, 1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = (
                self.affine_alpha * grouped_points + self.affine_beta
            )

        new_points = torch.cat(
            [
                grouped_points,
                new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1),
            ],
            dim=-1,
        )
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(
        self, in_channels, out_channels, bias=True, activation="relu"
    ):
        super().__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act,
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(
        self,
        channel,
        groups=1,
        res_expansion=1.0,
        bias=True,
        activation="relu",
    ):
        super().__init__()
        self.act = get_activation(activation)
        hidden = int(channel * res_expansion)
        self.net1 = nn.Sequential(
            nn.Conv1d(channel, hidden, 1, groups=groups, bias=bias),
            nn.BatchNorm1d(hidden),
            self.act,
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(hidden, channel, 1, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(channel, channel, 1, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(hidden, channel, 1, bias=bias),
                nn.BatchNorm1d(channel),
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    """Grouped features (B, G, K, D) to anchor features (B, D', G)."""

    def __init__(
        self,
        channels,
        out_channels,
        blocks=1,
        groups=1,
        res_expansion=1,
        bias=True,
        activation="relu",
        use_xyz=True,
    ):
        super().__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(
            in_channels, out_channels, bias=bias, activation=activation
        )
        self.operation = nn.Sequential(
            *[
                ConvBNReLURes1D(
                    out_channels,
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
                for _ in range(blocks)
            ]
        )

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2).reshape(-1, d, s)
        x = self.operation(self.transfer(x))
        x = F.adaptive_max_pool1d(x, 1).view(x.shape[0], -1)
        return x.reshape(b, n, -1).permute(0, 2, 1)


class PosExtraction(nn.Module):
    def __init__(
        self,
        channels,
        blocks=1,
        groups=1,
        res_expansion=1,
        bias=True,
        activation="relu",
    ):
        super().__init__()
        self.operation = nn.Sequential(
            *[
                ConvBNReLURes1D(
                    channels,
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
                for _ in range(blocks)
            ]
        )

    def forward(self, x):
        return self.operation(x)


class FeaturePropagation(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        blocks=1,
        groups=1,
        res_expansion=1.0,
        bias=True,
        activation="relu",
    ):
        super().__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, bias=bias)
        self.extraction = PosExtraction(
            out_channel,
            blocks,
            groups=groups,
            res_expansion=res_expansion,
            bias=bias,
            activation=activation,
        )

    def forward(self, xyz1, xyz2, points1, points2):
        """Upsample points2 (B, D2, S) at xyz2 to xyz1 and fuse points1."""
        points2 = points2.permute(0, 2, 1)
        B, N, _ = xyz1.shape
        S = xyz2.shape[1]
        if S == 1:
            interpolated = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            weight = dist_recip / dist_recip.sum(dim=2, keepdim=True)
            interpolated = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )
        if points1 is not None:
            new_points = torch.cat(
                [points1.permute(0, 2, 1), interpolated], dim=-1
            )
        else:
            new_points = interpolated
        new_points = self.fuse(new_points.permute(0, 2, 1))
        return self.extraction(new_points)


class PointMLP(LandmarkNet):
    """PointMLP encoder-decoder with a CHaR variant.

    The presence head reads the last encoder stage pooled with
    ``head_pooling``: ``max`` or ``max_avg`` (max and average concatenated).
    """

    def __init__(
        self,
        variant: str = "charm",
        num_landmarks: int = 80,
        num_teeth: int = 16,
        points: int = 10001,
        embed_dim: int = 64,
        groups: int = 1,
        res_expansion: float = 1.0,
        activation: str = "relu",
        bias: bool = False,
        use_xyz: bool = True,
        normalize: str = "anchor",
        dim_expansion: tuple[int, ...] = (2, 2, 2, 2),
        pre_blocks: tuple[int, ...] = (1, 1, 1, 1),
        pos_blocks: tuple[int, ...] = (1, 1, 1, 1),
        k_neighbors: tuple[int, ...] = (32, 32, 32, 32),
        reducers: tuple[int, ...] = (4, 4, 4, 4),
        de_dims: tuple[int, ...] = (512, 256, 128, 128),
        de_blocks: tuple[int, ...] = (1, 1, 1, 1),
        gmp_dim: int = 64,
        dropout: float = 0.5,
        head_pooling: str = "max_avg",
    ):
        super().__init__(variant, num_landmarks, num_teeth)
        stages = len(pre_blocks)
        if not (
            stages
            == len(k_neighbors)
            == len(reducers)
            == len(pos_blocks)
            == len(dim_expansion)
        ):
            raise ValueError("Every per-stage argument needs one value")
        if head_pooling not in ("max", "max_avg"):
            raise ValueError("head_pooling must be max or max_avg")
        self.stages = stages
        self.head_pooling = head_pooling
        self.embedding = ConvBNReLU1D(
            6, embed_dim, bias=bias, activation=activation
        )
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = points
        en_dims = [last_channel]
        for i in range(stages):
            out_channel = last_channel * dim_expansion[i]
            anchor_points = anchor_points // reducers[i]
            self.local_grouper_list.append(
                LocalGrouper(
                    last_channel,
                    anchor_points,
                    k_neighbors[i],
                    use_xyz,
                    normalize,
                )
            )
            self.pre_blocks_list.append(
                PreExtraction(
                    last_channel,
                    out_channel,
                    pre_blocks[i],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                    use_xyz=use_xyz,
                )
            )
            self.pos_blocks_list.append(
                PosExtraction(
                    out_channel,
                    pos_blocks[i],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )
            last_channel = out_channel
            en_dims.append(last_channel)
        cls_channels = last_channel

        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims = [en_dims[0], *de_dims]
        if not len(en_dims) == len(de_dims) == len(de_blocks) + 1:
            raise ValueError("de_dims and de_blocks need one value per stage")
        for i in range(len(en_dims) - 1):
            self.decode_list.append(
                FeaturePropagation(
                    de_dims[i] + en_dims[i + 1],
                    de_dims[i + 1],
                    blocks=de_blocks[i],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )

        self.act = get_activation(activation)
        self.gmp_map_list = nn.ModuleList(
            [
                ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation)
                for en_dim in en_dims
            ]
        )
        self.gmp_map_end = ConvBNReLU1D(
            gmp_dim * len(en_dims), gmp_dim, bias=bias, activation=activation
        )
        self.heat_regressor = nn.Sequential(
            nn.Conv1d(gmp_dim + de_dims[-1], 512, 1, bias=False),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(dropout),
            nn.Conv1d(512, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            self.act,
            nn.Dropout(dropout),
            nn.Conv1d(128, num_landmarks, 1),
            nn.Sigmoid(),
        )
        if self.has_presence_head:
            factor = 2 if head_pooling == "max_avg" else 1
            self.cls_head = presence_head(
                cls_channels * factor, (512, 256), num_teeth, dropout
            )

    def forward(self, points: torch.Tensor, normals: torch.Tensor):
        xyz = points
        x = self.embedding(
            torch.cat([points, normals], dim=-1).transpose(1, 2)
        )

        xyz_list = [xyz]
        x_list = [x]
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
            xyz_list.append(xyz)
            x_list.append(x)
        x_cls = x_list[-1]

        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        for i, decode in enumerate(self.decode_list):
            x = decode(xyz_list[i + 1], xyz_list[i], x_list[i + 1], x)

        gmp_list = [
            F.adaptive_max_pool1d(gmp_map(features), 1)
            for gmp_map, features in zip(self.gmp_map_list, x_list)
        ]
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1))
        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]])], dim=1)
        heatmaps = self.heat_regressor(x)

        logits = None
        if self.has_presence_head:
            pooled = x_cls.max(dim=-1)[0]
            if self.head_pooling == "max_avg":
                pooled = torch.cat((pooled, x_cls.mean(dim=-1)), dim=1)
            logits = self.cls_head(pooled)
        return self.finish(heatmaps, logits)
