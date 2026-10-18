"""Point Transformer landmark network.

Zhao et al., "Point Transformer", ICCV 2021. Adapted from
https://github.com/POSTECH-CVLab/point-transformer.

Internally the batch is packed as in the reference implementation: points
of shape (n, 3) with cumulative counts ``o`` of shape (b).
"""

from __future__ import annotations

import torch
from torch import nn

from charm.models.char import LandmarkNet, presence_head
from charm.models.ops import furthest_sampling, interpolation, query_and_group


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, mid_planes // share_planes),
            nn.BatchNorm1d(mid_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k = query_and_group(self.nsample, p, p, x_k, o, o, use_xyz=True)
        x_v = query_and_group(self.nsample, p, p, x_v, o, o, use_xyz=False)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p):
            if i == 1:
                p_r = layer(p_r.transpose(1, 2).contiguous())
                p_r = p_r.transpose(1, 2).contiguous()
            else:
                p_r = layer(p_r)
        w = (
            x_k
            - x_q.unsqueeze(1)
            + p_r.view(
                p_r.shape[0],
                p_r.shape[1],
                self.out_planes // self.mid_planes,
                self.mid_planes,
            ).sum(2)
        )
        for i, layer in enumerate(self.linear_w):
            if i % 3 == 0:
                w = layer(w.transpose(1, 2).contiguous())
                w = w.transpose(1, 2).contiguous()
            else:
                w = layer(w)
        w = self.softmax(w)
        n, nsample, c = x_v.shape
        s = self.share_planes
        x = (x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)
        return x.sum(1).view(n, c)


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        if self.stride == 1:
            return [p, self.relu(self.bn(self.linear(x))), o]
        counts = torch.diff(o, prepend=o.new_zeros(1)) // self.stride
        n_o = torch.cumsum(counts, 0).to(torch.int32)
        idx = furthest_sampling(p, o, n_o)
        n_p = p[idx, :]
        x = query_and_group(self.nsample, p, n_p, x, o, n_o, use_xyz=True)
        x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))
        x = self.pool(x).squeeze(-1)
        return [n_p, x, n_o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True)
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1
            x_tmp = []
            start = 0
            for end in o.tolist():
                x_b = x[start:end, :]
                count = end - start
                x_global = self.linear2(x_b.sum(0, True) / count)
                x_tmp.append(torch.cat((x_b, x_global.repeat(count, 1)), 1))
                start = end
            return self.linear1(torch.cat(x_tmp, 0))
        p1, x1, o1 = pxo1
        p2, x2, o2 = pxo2
        return self.linear1(x1) + interpolation(
            p2, p1, self.linear2(x2), o2, o1
        )


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(
            planes, planes, share_planes, nsample
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x = self.relu(x + identity)
        return [p, x, o]


class PointTransformer(LandmarkNet):
    """Point Transformer U-Net with a CHaR variant.

    The presence head reads the per-scan mean of the deepest encoder
    features.
    """

    def __init__(
        self,
        variant: str = "charm",
        num_landmarks: int = 80,
        num_teeth: int = 16,
        blocks: tuple[int, ...] = (2, 2, 2, 2, 2),
        width: int = 64,
        nsample: tuple[int, ...] = (8, 16, 16, 16, 16),
        in_channels: int = 6,
        dec_local_aggr: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__(variant, num_landmarks, num_teeth)
        if len(blocks) != 5 or len(nsample) != 5:
            raise ValueError("blocks and nsample need five stages")
        self.in_planes = in_channels
        self.planes = [width * 2**i for i in range(len(blocks))]
        share_planes = 8
        stride = [1, 4, 4, 4, 4]
        self.dec_local_aggr = dec_local_aggr
        for i in range(5):
            setattr(
                self,
                f"enc{i + 1}",
                self._make_enc(
                    self.planes[i],
                    blocks[i],
                    share_planes,
                    stride=stride[i],
                    nsample=nsample[i],
                ),
            )
        for i in reversed(range(5)):
            setattr(
                self,
                f"dec{i + 1}",
                self._make_dec(
                    self.planes[i],
                    2,
                    share_planes,
                    nsample[i],
                    is_head=i == 4,
                ),
            )
        self.heat_regressor = nn.Sequential(
            nn.Linear(self.planes[0], self.planes[0], bias=False),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.planes[0], num_landmarks),
            nn.Sigmoid(),
        )
        if self.has_presence_head:
            self.cls_head = presence_head(
                self.planes[-1], (128,), num_teeth, dropout
            )

    def _make_enc(self, planes, blocks, share_planes, stride, nsample):
        layers = [TransitionDown(self.in_planes, planes, stride, nsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(
                PointTransformerBlock(
                    self.in_planes, self.in_planes, share_planes, nsample
                )
            )
        return nn.Sequential(*layers)

    def _make_dec(self, planes, blocks, share_planes, nsample, is_head):
        layers = [TransitionUp(self.in_planes, None if is_head else planes)]
        self.in_planes = planes
        if self.dec_local_aggr:
            for _ in range(1, blocks):
                layers.append(
                    PointTransformerBlock(
                        self.in_planes, self.in_planes, share_planes, nsample
                    )
                )
        return nn.Sequential(*layers)

    def forward(self, points: torch.Tensor, normals: torch.Tensor):
        B, N, _ = points.shape
        p0 = points.reshape(-1, 3).contiguous()
        x0 = torch.cat((points, normals), -1).reshape(B * N, -1).contiguous()
        o0 = torch.arange(1, B + 1, device=points.device, dtype=torch.int32)
        o0 = o0 * N

        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        deepest = x5.view(B, -1, x5.shape[-1])

        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[
            1
        ]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[
            1
        ]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[
            1
        ]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[
            1
        ]
        heatmaps = self.heat_regressor(x1).view(B, N, -1).transpose(1, 2)

        logits = None
        if self.has_presence_head:
            logits = self.cls_head(deepest.mean(dim=1))
        return self.finish(heatmaps.contiguous(), logits)
