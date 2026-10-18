"""Convolution, normalization and activation builders.

Adapted from PointNeXt (https://github.com/guochengqian/PointNeXt, MIT).
"""

from __future__ import annotations

import copy

from torch import nn

_ACT_LAYER = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

_NORM_LAYER = {
    "bn1d": nn.BatchNorm1d,
    "bn2d": nn.BatchNorm2d,
    "in1d": nn.InstanceNorm1d,
    "in2d": nn.InstanceNorm2d,
}


def _as_dict(args, key: str) -> dict | None:
    if args is None:
        return None
    if isinstance(args, str):
        return {key: args}
    return dict(copy.deepcopy(args))


def create_act(act_args) -> nn.Module | None:
    """Activation layer from ``{"act": name, **kwargs}`` or a name."""
    act_args = _as_dict(act_args, "act")
    if act_args is None:
        return None
    act = act_args.pop("act", None)
    if act is None:
        return None
    act = act.lower()
    if act not in _ACT_LAYER:
        raise ValueError(f"Unsupported activation {act}")
    inplace = act_args.pop("inplace", True)
    if act == "gelu":
        return nn.GELU(**act_args)
    return _ACT_LAYER[act](inplace=inplace, **act_args)


def create_norm(norm_args, channels: int, dimension: str) -> nn.Module | None:
    """Normalization layer from ``{"norm": name, **kwargs}`` or a name."""
    norm_args = _as_dict(norm_args, "norm")
    if norm_args is None:
        return None
    norm = norm_args.pop("norm", None)
    if norm is None:
        return None
    norm = norm.lower()
    if dimension not in norm:
        norm += dimension
    if norm not in _NORM_LAYER:
        raise ValueError(f"Unsupported normalization {norm}")
    return _NORM_LAYER[norm](channels, **norm_args)


def _create_convblock(
    conv_cls,
    dimension: str,
    in_channels: int,
    out_channels: int,
    norm_args=None,
    act_args=None,
    order: str = "conv-norm-act",
    bias: bool = True,
) -> nn.Sequential:
    if order != "conv-norm-act":
        raise NotImplementedError(f"Order {order} is not supported")
    norm_layer = create_norm(norm_args, out_channels, dimension)
    if norm_layer is not None:
        bias = False
    layers = [conv_cls(in_channels, out_channels, 1, bias=bias)]
    if norm_layer is not None:
        layers.append(norm_layer)
    act_layer = create_act(act_args)
    if act_layer is not None:
        layers.append(act_layer)
    return nn.Sequential(*layers)


def create_convblock1d(in_channels, out_channels, **kwargs) -> nn.Sequential:
    """Pointwise Conv1d followed by optional normalization and activation."""
    return _create_convblock(
        nn.Conv1d, "1d", in_channels, out_channels, **kwargs
    )


def create_convblock2d(in_channels, out_channels, **kwargs) -> nn.Sequential:
    """Pointwise Conv2d followed by optional normalization and activation."""
    return _create_convblock(
        nn.Conv2d, "2d", in_channels, out_channels, **kwargs
    )
