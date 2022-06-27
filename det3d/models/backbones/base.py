import torch
import numpy as np
from torch import nn

try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
except:
    import spconv
    from spconv import SparseConv3d, SubMConv3d

from timm.models.layers import DropPath
from ..utils import build_norm_layer


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

def conv3x3(in_planes, out_planes, stride=1, dilation=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        dilation=dilation,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

def conv2D3x3(in_planes, out_planes, stride=1, dilation=1, indice_key=None, bias=True):
    """3x3 convolution with padding to keep the same input and output"""
    assert stride >= 1
    padding = dilation
    if stride == 1:
        return spconv.SubMConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )
    else:
        return spconv.SparseConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )

def conv2D1x1(in_planes, out_planes, bias=False):
    """1x1 convolution"""
    return spconv.SubMConv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            bias=bias,
            # indice_key=indice_key,
        )


class Sparse2DBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        indice_key=None,
    ):
        super(Sparse2DBasicBlock, self).__init__()
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = spconv.SparseSequential(
            conv2D3x3(planes, planes, stride, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv2 = spconv.SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class Sparse2DBasicBlockV(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        indice_key=None,
    ):
        super(Sparse2DBasicBlockV, self).__init__()
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv0 = spconv.SparseSequential(
            conv2D3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv1 = spconv.SparseSequential(
            conv2D3x3(planes, planes, stride, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv2 = spconv.SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = replace_feature(x, self.relu(x.features))
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class Dense2DBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
    ):
        super(Dense2DBasicBlock, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out

class Dense2DBasicBlockV(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
    ):
        super(Dense2DBasicBlockV, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        self.conv0 = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)

        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out
