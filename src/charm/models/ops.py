"""Point cloud operators with optional CUDA kernels.

The CUDA extensions ``pointnet2_batch_cuda`` and ``pointops_cuda`` (the
``cuda-ops`` extra) are used for CUDA tensors when installed. Otherwise the
pure PyTorch implementations below are used; they return the same indices,
in the same order, as the kernels.
"""

from __future__ import annotations

import torch
from torch.autograd import Function

try:
    import pointnet2_batch_cuda as _pn2_cuda
except ImportError:
    _pn2_cuda = None

try:
    import pointops_cuda as _pointops_cuda
except ImportError:
    _pointops_cuda = None

_CHUNK = 1024


def cuda_ops_available() -> bool:
    """Whether both CUDA extensions are installed."""
    return _pn2_cuda is not None and _pointops_cuda is not None


def _on_cuda(extension, *tensors: torch.Tensor) -> bool:
    return extension is not None and all(t.is_cuda for t in tensors)


def _square_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared distances between (..., n, 3) and (..., m, 3)."""
    return ((a.unsqueeze(-2) - b.unsqueeze(-3)) ** 2).sum(-1)


# Dense batches: xyz (B, N, 3), features (B, C, N).


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Indices (B, npoint) of farthest point sampling starting at point 0."""
    xyz = xyz.contiguous()
    B, N, _ = xyz.shape
    if _on_cuda(_pn2_cuda, xyz):
        output = torch.zeros(B, npoint, device=xyz.device, dtype=torch.int32)
        temp = torch.full((B, N), 1e10, device=xyz.device)
        _pn2_cuda.furthest_point_sampling_wrapper(
            B, N, npoint, xyz.float(), temp, output
        )
        return output.long()
    with torch.no_grad():
        idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
        dist = torch.full((B, N), 1e10, device=xyz.device)
        farthest = torch.zeros(B, dtype=torch.long, device=xyz.device)
        batch = torch.arange(B, device=xyz.device)
        for i in range(npoint):
            idx[:, i] = farthest
            centroid = xyz[batch, farthest].unsqueeze(1)
            dist = torch.minimum(dist, ((xyz - centroid) ** 2).sum(-1))
            farthest = dist.argmax(-1)
    return idx


def ball_query(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """First ``nsample`` points (by index) within ``radius`` of each query.

    Missing neighbours repeat the first one. Returns (B, M, nsample).
    """
    xyz, new_xyz = xyz.contiguous(), new_xyz.contiguous()
    B, N, _ = xyz.shape
    M = new_xyz.shape[1]
    if _on_cuda(_pn2_cuda, xyz, new_xyz):
        idx = torch.zeros(B, M, nsample, device=xyz.device, dtype=torch.int32)
        _pn2_cuda.ball_query_wrapper(
            B, N, M, radius, nsample, new_xyz.float(), xyz.float(), idx
        )
        return idx.long()
    with torch.no_grad():
        chunks = []
        arange = torch.arange(N, device=xyz.device)
        for start in range(0, M, _CHUNK):
            d2 = _square_distance(new_xyz[:, start : start + _CHUNK], xyz)
            candidates = torch.where(d2 < radius**2, arange, N)
            k = min(nsample, N)
            nearest = candidates.topk(k, dim=-1, largest=False).values
            if k < nsample:
                pad = nearest[..., :1].expand(-1, -1, nsample - k)
                nearest = torch.cat((nearest, pad), dim=-1)
            first = nearest[..., :1].expand_as(nearest)
            nearest = torch.where(nearest == N, first, nearest)
            chunks.append(torch.where(nearest == N, 0, nearest))
    return torch.cat(chunks, dim=1)


class _GroupingCUDA(Function):
    @staticmethod
    def forward(ctx, features, idx):
        B, C, N = features.shape
        _, M, K = idx.shape
        output = torch.zeros(B, C, M, K, device=features.device)
        _pn2_cuda.group_points_wrapper(B, C, N, M, K, features, idx, output)
        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, N = ctx.for_backwards
        B, C, M, K = grad_out.shape
        grad = torch.zeros(B, C, N, device=grad_out.device)
        _pn2_cuda.group_points_grad_wrapper(
            B, C, N, M, K, grad_out.contiguous(), idx, grad
        )
        return grad, None


def grouping_operation(features: torch.Tensor, idx: torch.Tensor):
    """Gather features (B, C, N) at idx (B, M, K) into (B, C, M, K)."""
    if _on_cuda(_pn2_cuda, features) and features.dtype == torch.float32:
        return _GroupingCUDA.apply(
            features.contiguous(), idx.int().contiguous()
        )
    B, C, _ = features.shape
    flat = idx.reshape(B, 1, -1).long().expand(-1, C, -1)
    return features.gather(2, flat).reshape(B, C, *idx.shape[1:])


def three_nn(unknown: torch.Tensor, known: torch.Tensor):
    """Distances and indices (B, N, 3) of the three nearest known points."""
    unknown, known = unknown.contiguous(), known.contiguous()
    B, N, _ = unknown.shape
    if _on_cuda(_pn2_cuda, unknown, known):
        dist2 = torch.zeros(B, N, 3, device=unknown.device)
        idx = torch.zeros(B, N, 3, device=unknown.device, dtype=torch.int32)
        _pn2_cuda.three_nn_wrapper(
            B, N, known.shape[1], unknown.float(), known.float(), dist2, idx
        )
        return torch.sqrt(dist2), idx.long()
    dists, indices = [], []
    with torch.no_grad():
        for start in range(0, N, _CHUNK):
            d2 = _square_distance(unknown[:, start : start + _CHUNK], known)
            values, idx = d2.topk(3, dim=-1, largest=False)
            dists.append(values)
            indices.append(idx)
    return torch.sqrt(torch.cat(dists, 1)), torch.cat(indices, 1)


def three_interpolation(
    unknown_xyz: torch.Tensor,
    known_xyz: torch.Tensor,
    known_feat: torch.Tensor,
) -> torch.Tensor:
    """Inverse-distance interpolation of known_feat (B, C, M) to (B, C, N)."""
    dist, idx = three_nn(unknown_xyz, known_xyz)
    dist_recip = 1.0 / (dist + 1e-8)
    weight = dist_recip / dist_recip.sum(dim=2, keepdim=True)
    neighbours = grouping_operation(known_feat, idx)
    return (neighbours * weight.unsqueeze(1)).sum(-1)


# Packed batches: xyz (n, 3) with cumulative point counts ``offset`` (b).


def _segments(offset: torch.Tensor) -> list[tuple[int, int]]:
    ends = offset.tolist()
    return list(zip([0] + ends[:-1], ends))


def knn_query(
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    offset: torch.Tensor,
    new_offset: torch.Tensor,
):
    """Indices and distances (m, nsample) of the nearest points, ascending."""
    xyz, new_xyz = xyz.contiguous(), new_xyz.contiguous()
    m = new_xyz.shape[0]
    if _on_cuda(_pointops_cuda, xyz, new_xyz):
        idx = torch.zeros(m, nsample, device=xyz.device, dtype=torch.int32)
        dist2 = torch.zeros(m, nsample, device=xyz.device)
        _pointops_cuda.knnquery_cuda(
            m,
            nsample,
            xyz.float(),
            new_xyz.float(),
            offset.int(),
            new_offset.int(),
            idx,
            dist2,
        )
        return idx.long(), torch.sqrt(dist2)
    idx_out, dist_out = [], []
    with torch.no_grad():
        segments = zip(_segments(offset), _segments(new_offset))
        for (s, e), (new_s, new_e) in segments:
            support = xyz[s:e]
            for start in range(new_s, new_e, _CHUNK):
                query = new_xyz[start : min(start + _CHUNK, new_e)]
                d2 = _square_distance(query, support)
                k = min(nsample, support.shape[0])
                values, idx = d2.topk(k, dim=-1, largest=False)
                if k < nsample:
                    pad = (query.shape[0], nsample - k)
                    idx = torch.cat((idx, idx.new_zeros(pad) - s), dim=1)
                    values = torch.cat((values, values.new_full(pad, 1e10)), 1)
                idx_out.append(idx + s)
                dist_out.append(torch.sqrt(values))
    return torch.cat(idx_out), torch.cat(dist_out)


def furthest_sampling(
    xyz: torch.Tensor, offset: torch.Tensor, new_offset: torch.Tensor
) -> torch.Tensor:
    """Farthest point sampling per segment, starting at its first point."""
    xyz = xyz.contiguous()
    if _on_cuda(_pointops_cuda, xyz):
        b = offset.shape[0]
        n_max = int(torch.diff(offset, prepend=offset.new_zeros(1)).max())
        idx = torch.zeros(
            int(new_offset[-1]), device=xyz.device, dtype=torch.int32
        )
        tmp = torch.full((xyz.shape[0],), 1e10, device=xyz.device)
        _pointops_cuda.furthestsampling_cuda(
            b, n_max, xyz.float(), offset.int(), new_offset.int(), tmp, idx
        )
        return idx.long()
    out = []
    segments = zip(_segments(offset), _segments(new_offset))
    for (s, e), (new_s, new_e) in segments:
        local = furthest_point_sample(xyz[s:e].unsqueeze(0), new_e - new_s)
        out.append(local[0] + s)
    return torch.cat(out)


def query_and_group(
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    feat: torch.Tensor,
    offset: torch.Tensor,
    new_offset: torch.Tensor,
    use_xyz: bool = True,
) -> torch.Tensor:
    """Features of the k nearest neighbours, (m, nsample, [3 +] c)."""
    idx, _ = knn_query(nsample, xyz, new_xyz, offset, new_offset)
    m, c = new_xyz.shape[0], feat.shape[1]
    flat = idx.view(-1)
    grouped_feat = feat[flat].view(m, nsample, c)
    if not use_xyz:
        return grouped_feat
    grouped_xyz = xyz[flat].view(m, nsample, 3) - new_xyz.unsqueeze(1)
    return torch.cat((grouped_xyz, grouped_feat), -1)


def interpolation(
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    feat: torch.Tensor,
    offset: torch.Tensor,
    new_offset: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    """Inverse-distance interpolation of feat (m, c) at new_xyz (n, 3)."""
    idx, dist = knn_query(k, xyz, new_xyz, offset, new_offset)
    dist_recip = 1.0 / (dist + 1e-8)
    weight = dist_recip / dist_recip.sum(dim=1, keepdim=True)
    new_feat = torch.zeros(
        new_xyz.shape[0], feat.shape[1], device=feat.device, dtype=feat.dtype
    )
    for i in range(k):
        new_feat = new_feat + feat[idx[:, i]] * weight[:, i].unsqueeze(-1)
    return new_feat
