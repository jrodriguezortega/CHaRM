"""PointNet++ (single-scale grouping) landmark network.

Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a
Metric Space", NeurIPS 2017. Pure PyTorch implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from charm.models.char import LandmarkNet, presence_head


def get_dists(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """Euclidean distances (B, M, N) between (B, M, C) and (B, N, C)."""
    B, M, _ = points1.shape
    N = points2.shape[1]
    dists = (points1**2).sum(-1).view(B, M, 1) + (points2**2).sum(-1).view(
        B, 1, N
    )
    dists = dists - 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.full_like(dists, 1e-7), dists)
    return torch.sqrt(dists).float()


def gather_points(points: torch.Tensor, inds: torch.Tensor) -> torch.Tensor:
    """Select (B, M[, K], C) from points (B, N, C) by inds (B, M[, K])."""
    B = points.shape[0]
    view = [B] + [1] * (inds.dim() - 1)
    batch = torch.arange(B, device=points.device).view(view).expand_as(inds)
    return points[batch, inds, :]


def three_interpolate(xyz1, xyz2, points2):
    """Interpolate points2 (B, N2, C) at xyz1 from its three nearest."""
    dists = get_dists(xyz1, xyz2)
    k = min(3, dists.shape[-1])
    dists, inds = torch.topk(dists, k, dim=-1, largest=False)
    inversed = 1.0 / (dists + 1e-8)
    weight = inversed / inversed.sum(dim=-1, keepdim=True)
    return (weight.unsqueeze(-1) * gather_points(points2, inds)).sum(dim=2)


def fps(xyz: torch.Tensor, num: int) -> torch.Tensor:
    """Farthest point sampling from a random first point, (B, num)."""
    batch, n, _ = xyz.shape
    dists = torch.full((batch, n), float("inf"), device=xyz.device)
    farthest = torch.randint(0, n, (batch,), device=xyz.device)
    batch_indices = torch.arange(batch, device=xyz.device)
    centroids = []
    for _ in range(num):
        centroids.append(farthest)
        current = xyz[batch_indices, farthest, :].unsqueeze(1)
        dists = torch.minimum(dists, ((xyz - current) ** 2).sum(-1))
        farthest = dists.max(dim=1)[1]
    return torch.stack(centroids).t()


def ball_query(xyz, new_xyz, radius: float, K: int) -> torch.Tensor:
    """First K points by index within radius, padded with the first."""
    B, N, _ = xyz.shape
    M = new_xyz.shape[1]
    inds = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, M, 1)
    dists = get_dists(new_xyz, xyz)
    inds = torch.where(dists > radius, N, inds)
    inds = torch.topk(inds, K, largest=False)[0]
    first = inds[:, :, 0:1].repeat(1, 1, K)
    inds[inds == N] = first[inds == N]
    return inds


def sample_and_group(xyz, points, M: int, radius: float, K: int):
    new_xyz = gather_points(xyz, fps(xyz, M))
    grouped_inds = ball_query(xyz, new_xyz, radius, K)
    grouped_xyz = gather_points(xyz, grouped_inds)
    grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
    grouped_points = gather_points(points, grouped_inds)
    new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), -1)
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    B, M, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=xyz.device)
    new_points = torch.cat([xyz.float(), points.float()], dim=2).unsqueeze(1)
    return new_xyz, new_points


def _conv_stack(in_channels: int, mlp: list[int], sep: str = ""):
    layers = nn.Sequential()
    for i, out_channels in enumerate(mlp):
        layers.add_module(
            f"Conv{sep}{i}",
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )
        layers.add_module(f"Bn{sep}{i}", nn.BatchNorm2d(out_channels))
        layers.add_module(f"Relu{sep}{i}", nn.ReLU())
        in_channels = out_channels
    return layers


class SetAbstraction(nn.Module):
    def __init__(self, M, radius, K, in_channels, mlp, group_all):
        super().__init__()
        self.M, self.radius, self.K = M, radius, K
        self.group_all = group_all
        self.backbone = _conv_stack(in_channels, mlp)

    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                xyz, points, self.M, self.radius, self.K
            )
        new_points = self.backbone(new_points.permute(0, 3, 2, 1))
        new_points = torch.max(new_points, dim=2)[0]
        return new_xyz, new_points.permute(0, 2, 1).contiguous()


class FeaturePropagation(nn.Module):
    def __init__(self, in_channels, mlp):
        super().__init__()
        self.backbone = _conv_stack(in_channels, mlp, sep="_")

    def forward(self, xyz1, xyz2, points1, points2):
        interpolated = three_interpolate(xyz1, xyz2, points2)
        features = torch.cat([interpolated, points1], dim=-1)
        features = features.permute(0, 2, 1).unsqueeze(-1)
        new_points = self.backbone(features).squeeze(-1)
        return new_points.permute(0, 2, 1).contiguous()


class PointNet2(LandmarkNet):
    """PointNet++ SSG segmentation network with a CHaR variant.

    The presence head reads the global feature of the last set abstraction.
    """

    def __init__(
        self,
        variant: str = "charm",
        num_landmarks: int = 80,
        num_teeth: int = 16,
        in_channels: int = 6,
        dropout: float = 0.5,
    ):
        super().__init__(variant, num_landmarks, num_teeth)
        self.pt_sa1 = SetAbstraction(
            512, 0.2, 32, in_channels, [64, 64, 128], False
        )
        self.pt_sa2 = SetAbstraction(128, 0.4, 64, 131, [128, 128, 256], False)
        self.pt_sa3 = SetAbstraction(
            None, None, None, 259, [256, 512, 1024], True
        )
        self.pt_fp1 = FeaturePropagation(1024 + 256, [256, 256])
        self.pt_fp2 = FeaturePropagation(256 + 128, [256, 128])
        self.pt_fp3 = FeaturePropagation(128 + in_channels, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout)
        self.cls = nn.Conv1d(128, num_landmarks, 1)
        if self.has_presence_head:
            self.cls_layers = presence_head(
                1024, (512, 256), num_teeth, dropout
            )

    def forward(self, points: torch.Tensor, normals: torch.Tensor):
        B = points.shape[0]
        l1_xyz, l1_points = self.pt_sa1(points, normals)
        l2_xyz, l2_points = self.pt_sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.pt_sa3(l2_xyz, l2_points)

        l2_points = self.pt_fp1(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.pt_fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.pt_fp3(
            points, l1_xyz, torch.cat([normals, points], dim=-1), l1_points
        )

        net = l0_points.permute(0, 2, 1).contiguous()
        net = self.dropout1(F.relu(self.bn1(self.conv1(net))))
        heatmaps = torch.sigmoid(self.cls(net))

        logits = None
        if self.has_presence_head:
            logits = self.cls_layers(l3_points.view(B, 1024))
        return self.finish(heatmaps, logits)
