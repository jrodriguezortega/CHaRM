import pytest
import torch

from charm.constants import NUM_LANDMARKS, NUM_TEETH
from charm.models import (
    DGCNN,
    VARIANTS,
    PointMLP,
    PointNet2,
    PointNeXt,
    PointTransformer,
    combine_heatmaps,
)

NUM_POINTS = 2049

BACKBONES = {
    "pointnet2": lambda variant: PointNet2(variant),
    "dgcnn": lambda variant: DGCNN(variant, k=8),
    "pointmlp": lambda variant: PointMLP(
        variant,
        points=NUM_POINTS,
        k_neighbors=(8, 8, 8, 8),
        head_pooling="max" if variant == "postproc" else "max_avg",
    ),
    "pointnext": lambda variant: PointNeXt(variant),
    "pointtransformer": lambda variant: PointTransformer(variant),
}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backbone", sorted(BACKBONES))
def test_forward_shapes(backbone, variant):
    torch.manual_seed(0)
    net = BACKBONES[backbone](variant).eval()
    points = torch.rand(2, NUM_POINTS, 3) - 0.5
    points[:, -1] = torch.tensor([0.0, 0.5, 0.0])
    normals = torch.nn.functional.normalize(torch.randn(2, NUM_POINTS, 3))

    with torch.no_grad():
        heatmaps, presence = net(points, normals)

    assert heatmaps.shape == (2, NUM_LANDMARKS, NUM_POINTS)
    assert torch.isfinite(heatmaps).all()
    if variant == "base":
        assert presence is None
    else:
        assert presence.shape == (2, NUM_TEETH)
        assert ((presence >= 0) & (presence <= 1)).all()


def test_backward_pointmlp_charm():
    net = BACKBONES["pointmlp"]("charm")
    points = torch.rand(2, NUM_POINTS, 3) - 0.5
    normals = torch.nn.functional.normalize(torch.randn(2, NUM_POINTS, 3))
    heatmaps, presence = net(points, normals)
    (heatmaps.sum() + presence.sum()).backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_combine_heatmaps():
    heatmaps = torch.rand(1, NUM_LANDMARKS, 7)
    presence = torch.rand(1, NUM_TEETH)
    combined = combine_heatmaps(heatmaps, presence)

    p = presence.repeat_interleave(NUM_LANDMARKS // NUM_TEETH, dim=1)
    torch.testing.assert_close(
        combined[..., :-1], heatmaps[..., :-1] * p[..., None]
    )
    torch.testing.assert_close(combined[..., -1], heatmaps[..., -1] * (1 - p))


def test_invalid_variant():
    with pytest.raises(ValueError):
        PointNet2("unknown")
