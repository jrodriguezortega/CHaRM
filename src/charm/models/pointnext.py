"""PointNeXt landmark network.

Qian et al., "PointNeXt: Revisiting PointNet++ with Improved Training and
Scaling Strategies", NeurIPS 2022. Adapted from
https://github.com/guochengqian/PointNeXt (MIT).
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from charm.models.blocks import (
    create_act,
    create_convblock1d,
    create_convblock2d,
)
from charm.models.char import LandmarkNet, presence_head
from charm.models.ops import (
    ball_query,
    furthest_point_sample,
    grouping_operation,
    three_interpolation,
)

CHANNEL_MAP = {
    "fj": lambda x: x,
    "dp_fj": lambda x: 3 + x,
    "dp_fj_df": lambda x: x * 2 + 3,
}


def get_aggregation_features(dp, f, fj, feature_type="dp_fj"):
    if feature_type == "dp_fj":
        return torch.cat([dp, fj], 1)
    if feature_type == "dp_fj_df":
        return torch.cat([dp, fj, fj - f.unsqueeze(-1)], 1)
    if feature_type == "fj":
        return fj
    raise ValueError(f"Unsupported feature type {feature_type}")


def get_reduction_fn(reduction: str):
    reduction = "mean" if reduction.lower() == "avg" else reduction.lower()
    if reduction == "max":
        return lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    if reduction == "mean":
        return lambda x: torch.mean(x, dim=-1, keepdim=False)
    if reduction == "sum":
        return lambda x: torch.sum(x, dim=-1, keepdim=False)
    raise ValueError(f"Unsupported reduction {reduction}")


class QueryAndGroup(nn.Module):
    """Ball query grouping of relative positions and features."""

    def __init__(self, radius: float, nsample: int, normalize_dp=False):
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.normalize_dp = normalize_dp

    def forward(self, query_xyz, support_xyz, features=None):
        idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz)
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_dp:
            grouped_xyz = grouped_xyz / self.radius
        grouped_features = (
            grouping_operation(features, idx) if features is not None else None
        )
        return grouped_xyz, grouped_features


def create_grouper(group_args: dict) -> QueryAndGroup:
    group_args = dict(group_args)
    method = group_args.pop("NAME", "ballquery")
    if method != "ballquery":
        raise NotImplementedError(f"Grouper {method} is not implemented")
    return QueryAndGroup(
        group_args.pop("radius", 0.1),
        group_args.pop("nsample", 20),
        **group_args,
    )


class LocalAggregation(nn.Module):
    """Aggregate the features of the neighbours of every point."""

    def __init__(
        self,
        channels: list[int],
        norm_args=None,
        act_args=None,
        group_args=None,
        conv_args=None,
        feature_type="dp_fj",
        reduction="max",
        last_act=True,
    ):
        super().__init__()
        channels = list(channels)
        channels[0] = CHANNEL_MAP[feature_type](channels[0])
        convs = []
        for i in range(len(channels) - 1):
            is_last = i == len(channels) - 2
            convs.append(
                create_convblock2d(
                    channels[i],
                    channels[i + 1],
                    norm_args=norm_args,
                    act_args=None if is_last and not last_act else act_args,
                    **(conv_args or {}),
                )
            )
        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args)
        self.pool = get_reduction_fn(reduction)
        self.feature_type = feature_type

    def forward(self, pf):
        p, f = pf
        dp, fj = self.grouper(p, p, f)
        fj = get_aggregation_features(dp, f, fj, self.feature_type)
        return self.pool(self.convs(fj))


class SetAbstraction(nn.Module):
    """Set abstraction with optional residual connection.

    With ``is_head`` it is a pointwise stem on the input features.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        layers=1,
        stride=1,
        group_args=None,
        norm_args=None,
        act_args=None,
        conv_args=None,
        feature_type="dp_fj",
        use_res=False,
        is_head=False,
        **kwargs,
    ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        if not is_head and stride == 1:
            raise ValueError("Set abstraction without downsampling is unused")
        self.use_res = use_res and not is_head
        self.feature_type = feature_type

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * (layers - 1)
        channels.append(out_channels)
        if not is_head:
            channels[0] = CHANNEL_MAP[feature_type](channels[0])

        if self.use_res:
            self.skipconv = (
                create_convblock1d(in_channels, channels[-1])
                if in_channels != channels[-1]
                else nn.Identity()
            )
            self.act = create_act(act_args)

        create_conv = create_convblock1d if is_head else create_convblock2d
        convs = []
        for i in range(len(channels) - 1):
            is_last = i == len(channels) - 2
            convs.append(
                create_conv(
                    channels[i],
                    channels[i + 1],
                    norm_args=norm_args if not is_head else None,
                    act_args=(
                        None
                        if is_last and (self.use_res or is_head)
                        else act_args
                    ),
                    **(conv_args or {}),
                )
            )
        self.convs = nn.Sequential(*convs)
        if not is_head:
            self.grouper = create_grouper(group_args)

    def forward(self, pf):
        p, f = pf
        if self.is_head:
            return p, self.convs(f)
        idx = furthest_point_sample(p, p.shape[1] // self.stride)
        new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        fi = None
        if self.use_res or "df" in self.feature_type:
            fi = torch.gather(
                f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1)
            )
            if self.use_res:
                identity = self.skipconv(fi)
        dp, fj = self.grouper(new_p, p, f)
        fj = get_aggregation_features(dp, fi, fj, self.feature_type)
        f = torch.max(self.convs(fj), dim=-1, keepdim=False)[0]
        if self.use_res:
            f = self.act(f + identity)
        return new_p, f


class FeaturePropagation(nn.Module):
    """Interpolate coarse features to the fine points and fuse the skip."""

    def __init__(self, mlp, norm_args="bn1d", act_args="relu"):
        super().__init__()
        self.convs = nn.Sequential(
            *[
                create_convblock1d(
                    mlp[i], mlp[i + 1], norm_args=norm_args, act_args=act_args
                )
                for i in range(len(mlp) - 1)
            ]
        )

    def forward(self, pf1, pf2):
        p1, f1 = pf1
        p2, f2 = pf2
        interpolated = three_interpolation(p1, p2, f2)
        if f1 is not None:
            interpolated = torch.cat((f1, interpolated), dim=1)
        return self.convs(interpolated)


class InvResMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        norm_args=None,
        act_args=None,
        aggr_args=None,
        group_args=None,
        conv_args=None,
        expansion=1,
        use_res=True,
        num_posconvs=2,
        less_act=False,
    ):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation(
            [in_channels, in_channels],
            norm_args=norm_args,
            act_args=act_args if num_posconvs > 0 else None,
            group_args=group_args,
            conv_args=conv_args,
            **(aggr_args or {}),
        )
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        for i in range(len(channels) - 1):
            keep_act = i != len(channels) - 2 and not less_act
            pwconv.append(
                create_convblock1d(
                    channels[i],
                    channels[i + 1],
                    norm_args=norm_args,
                    act_args=act_args if keep_act else None,
                    **(conv_args or {}),
                )
            )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.pwconv(self.convs([p, f]))
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f = f + identity
        return [p, self.act(f)]


class PointNextEncoder(nn.Module):
    """PointNeXt encoder. Each stage is a set abstraction plus blocks."""

    def __init__(
        self,
        in_channels: int = 6,
        width: int = 128,
        blocks: tuple[int, ...] = (1, 2, 1, 1),
        strides: tuple[int, ...] = (1, 4, 4, 4),
        nsample: int = 32,
        radius: float = 0.1,
        aggr_args: dict | None = None,
        group_args: dict | None = None,
        norm_args: dict | None = None,
        act_args: dict | None = None,
        conv_args: dict | None = None,
        expansion: int = 4,
        sa_layers: int = 1,
        sa_use_res: bool = False,
        use_res: bool = True,
        radius_scaling: float = 2,
        nsample_scaling: float = 1,
    ):
        super().__init__()
        self.blocks = list(blocks)
        self.strides = list(strides)
        self.in_channels = in_channels
        self.aggr_args = dict(
            aggr_args or {"feature_type": "dp_fj", "reduction": "max"}
        )
        group_args = dict(
            group_args or {"NAME": "ballquery", "normalize_dp": True}
        )
        self.norm_args = dict(norm_args or {"norm": "bn"})
        self.act_args = dict(act_args or {"act": "relu"})
        self.conv_args = dict(conv_args or {"order": "conv-norm-act"})
        self.expansion = expansion
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = use_res
        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)

        channels = []
        for stride in self.strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        for i in range(len(self.blocks)):
            encoder.append(
                self._make_enc(
                    channels[i],
                    self.blocks[i],
                    stride=self.strides[i],
                    group_args=group_args,
                    radii=self.radii[i],
                    nsamples=self.nsample[i],
                    is_head=i == 0 and self.strides[i] == 1,
                )
            )
        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, scaling=1):
        param_list = []
        for i, stride in enumerate(self.strides):
            if stride == 1:
                param_list.append([param] * self.blocks[i])
            else:
                param_list.append(
                    [param] + [param * scaling] * (self.blocks[i] - 1)
                )
                param *= scaling
        return param_list

    def _make_enc(
        self, channels, blocks, stride, group_args, radii, nsamples, is_head
    ):
        def grouping(i):
            args = copy.deepcopy(group_args)
            args["radius"], args["nsample"] = radii[i], nsamples[i]
            return args

        layers = [
            SetAbstraction(
                self.in_channels,
                channels,
                self.sa_layers if not is_head else 1,
                stride,
                group_args=grouping(0),
                norm_args=self.norm_args,
                act_args=self.act_args,
                conv_args=self.conv_args,
                is_head=is_head,
                use_res=self.sa_use_res,
                **self.aggr_args,
            )
        ]
        self.in_channels = channels
        for i in range(1, blocks):
            layers.append(
                InvResMLP(
                    self.in_channels,
                    aggr_args=self.aggr_args,
                    norm_args=self.norm_args,
                    act_args=self.act_args,
                    group_args=grouping(i),
                    conv_args=self.conv_args,
                    expansion=self.expansion,
                    use_res=self.use_res,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, p0, f0):
        p, f = [p0], [f0]
        for stage in self.encoder:
            _p, _f = stage([p[-1], f[-1]])
            p.append(_p)
            f.append(_f)
        return p, f


class PointNextDecoder(nn.Module):
    def __init__(
        self,
        encoder_channel_list: list[int],
        decoder_layers: int = 2,
        decoder_stages: int = 3,
        in_channels: int = 3,
    ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = list(encoder_channel_list[:-1])
        if len(skip_channels) < decoder_stages:
            skip_channels.insert(0, in_channels)
        fp_channels = encoder_channel_list[:decoder_stages]
        n_stages = len(fp_channels)
        decoder = [[] for _ in range(n_stages)]
        for i in range(-1, -n_stages - 1, -1):
            decoder[i] = self._make_dec(skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_stages]

    def _make_dec(self, skip_channels, fp_channels):
        mlp = [skip_channels + self.in_channels]
        mlp += [fp_channels] * self.decoder_layers
        self.in_channels = fp_channels
        return nn.Sequential(FeaturePropagation(mlp))

    def forward(self, p, f):
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])]
            )[1]
        return f[-len(self.decoder) - 1]


class PointNeXt(LandmarkNet):
    """PointNeXt encoder-decoder with a CHaR variant.

    ``encoder`` and ``decoder`` hold the keyword arguments of
    ``PointNextEncoder`` and ``PointNextDecoder``. The presence head reads
    the deepest encoder features pooled with max.
    """

    def __init__(
        self,
        variant: str = "charm",
        num_landmarks: int = 80,
        num_teeth: int = 16,
        encoder: dict | None = None,
        decoder: dict | None = None,
        dropout: float = 0.5,
    ):
        super().__init__(variant, num_landmarks, num_teeth)
        self.encoder = PointNextEncoder(**dict(encoder or {}))
        self.decoder = PointNextDecoder(
            encoder_channel_list=self.encoder.channel_list,
            **dict(decoder or {}),
        )
        self.heat_regressor = nn.Sequential(
            nn.Conv1d(self.decoder.out_channels, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_landmarks, 1),
            nn.Sigmoid(),
        )
        if self.has_presence_head:
            self.cls_head = presence_head(
                self.encoder.channel_list[-1], (128,), num_teeth, dropout
            )

    def forward(self, points: torch.Tensor, normals: torch.Tensor):
        f0 = torch.cat([points, normals], dim=-1).transpose(1, 2).contiguous()
        p, f = self.encoder(points.contiguous(), f0)
        deepest = f[-1]
        heatmaps = self.heat_regressor(self.decoder(p, f).squeeze(-1))

        logits = None
        if self.has_presence_head:
            logits = self.cls_head(deepest.max(dim=-1)[0])
        return self.finish(heatmaps, logits)
