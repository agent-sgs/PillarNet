import torch
try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv2d, SparseMaxPool2d, SparseInverseConv2d
except:
    import spconv
    from spconv import ops
    from spconv import SparseConv2d, SparseMaxPool2d, SparseInverseConv2d

from torch import nn
from timm.models.layers import trunc_normal_

from ..registry import BACKBONES
from ..utils import build_norm_layer
from .base import Dense2DBasicBlock, Dense2DBasicBlockV

from det3d.ops.pillar_ops.pillar_modules import PillarMaxPooling


def post_block(in_channels, out_channels, kernel_size, stride=1, padding=1, norm_cfg=None, **kwargs):

    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        build_norm_layer(norm_cfg, out_channels)[1],
    )

    return m


def post_act_block(in_channels, out_channels, kernel_size, stride=1, padding=1, norm_cfg=None, **kwargs):

    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(),
    )

    return m

@BACKBONES.register_module
class DsPillarEncoderHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="PillarEncoder", **kwargs
    ):
        super(DsPillarEncoderHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling(
            mlps=[6 + num_input_features, 16*double],
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        convblock = post_block
        block = post_act_block
        self.conv1 = nn.Sequential(
            convblock(16 * double, 16 * double, 3, norm_cfg=norm_cfg),
            block(16 * double, 16 * double, 3, norm_cfg=norm_cfg, indice_key="subm1"),
        )

        self.conv2 = nn.Sequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16*double, 32*double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
            block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        )

        self.conv3 = nn.Sequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32*double, 64*double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        )

        self.conv4 = nn.Sequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64*double, 128*double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )
        self.conv5 = nn.Sequential(
            block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
            block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv1 = sp_tensor.dense()
        x_conv1 = self.conv1(x_conv1)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)

        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class DsPillarEncoder2xHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="PillarEncoder2xHA", **kwargs
    ):
        super(DsPillarEncoder2xHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling(
            mlps=[6 + num_input_features, 32*double],
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        convblock = post_block
        block = post_act_block

        self.conv2 = nn.Sequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            convblock(32 * double, 32 * double, 3, norm_cfg=norm_cfg),
            block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
            block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        )

        self.conv3 = nn.Sequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32*double, 64*double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        )

        self.conv4 = nn.Sequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64*double, 128*double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )
        self.conv5 = nn.Sequential(
            block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
            block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)

        # x_conv1 = self.conv1(sp_tensor)
        x_conv2 = sp_tensor.dense()
        x_conv2 = self.conv2(x_conv2)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)

        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )

@BACKBONES.register_module
class DsPillarEncoder4xHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="PillarEncoder", **kwargs
    ):
        super(DsPillarEncoder4xHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling(
            mlps=[6 + num_input_features, 64*double],
            bev_size=pillar_cfg['pool3']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        # self.conv1 = nn.Sequential(
        #     spconv.SubMConv2d(16*double, 16*double, 3, padding=1, bias=False, indice_key="subm1"),
        #     build_norm_layer(norm_cfg, 16*double)[1],
        #     block(16*double, 16*double, 3, norm_cfg=norm_cfg, indice_key="subm1"),
        # )

        # self.conv2 = nn.Sequential(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(16*double, 32*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        #     block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        # )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        convblock = post_block
        block = post_act_block

        self.conv3 = nn.Sequential(
            # [800, 704, 21] <- [400, 352, 11]
            convblock(64 * double, 64 * double, 3, norm_cfg=norm_cfg, padding=1),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        )

        self.conv4 = nn.Sequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64*double, 128*double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )
        self.conv5 = nn.Sequential(
            block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
            block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)

        # x_conv1 = self.conv1(sp_tensor)
        # x_conv2 = self.conv2(x_conv1)
        x_conv3 = sp_tensor.dense()
        x_conv3 = self.conv3(x_conv3)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)

        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class DsPillarEncoder8xHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="PillarEncoder", **kwargs
    ):
        super(DsPillarEncoder8xHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling(
            mlps=[6 + num_input_features, 128 * double],
            bev_size=pillar_cfg['pool4']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        # self.conv1 = nn.Sequential(
        #     spconv.SubMConv2d(16*double, 16*double, 3, padding=1, bias=False, indice_key="subm1"),
        #     build_norm_layer(norm_cfg, 16*double)[1],
        #     block(16*double, 16*double, 3, norm_cfg=norm_cfg, indice_key="subm1"),
        # )

        # self.conv2 = nn.Sequential(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(16*double, 32*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        #     block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        # )

        # self.conv3 = nn.Sequential(
        #     # [800, 704, 21] <- [400, 352, 11]
        #     block(32*double, 64*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        #     block(64 * double, 64 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        #     block(64 * double, 64 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        # )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        convblock = post_block
        block = post_act_block

        self.conv4 = nn.Sequential(
            # [400, 352, 11] <- [200, 176, 5]
            convblock(128 * double, 128 * double, 3, norm_cfg=norm_cfg, padding=1),
            block(128 * double, 128 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
            block(128 * double, 128 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )
        self.conv5 = nn.Sequential(
            block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
            block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)

        # x_conv1 = self.conv1(sp_tensor)
        # x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(sp_tensor)
        x_conv4 = sp_tensor.dense()
        x_conv4 = self.conv4(x_conv4)
        x_conv5 = self.conv5(x_conv4)

        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )
    

@BACKBONES.register_module
class DsMiddlePillarEncoderHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="DsMiddlePillarEncoderHA", **kwargs
    ):
        super(DsMiddlePillarEncoderHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling(
            mlps=[6 + num_input_features, 16 * double],
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        self.conv1 = nn.Sequential(
            Dense2DBasicBlockV(16 * double, 16 * double),
            Dense2DBasicBlock(16 * double, 16 * double),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16 * double, 32 * double, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 32 * double)[1],
            nn.ReLU(),
            Dense2DBasicBlock(32 * double, 32 * double),
            Dense2DBasicBlock(32 * double, 32 * double),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32 * double, 64 * double, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 64 * double)[1],
            nn.ReLU(),
            Dense2DBasicBlock(64 * double, 64 * double),
            Dense2DBasicBlock(64 * double, 64 * double),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64 * double, 128 * double, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 128 * double)[1],
            nn.ReLU(),
            Dense2DBasicBlock(128 * double, 128 * double),
            Dense2DBasicBlock(128 * double, 128 * double),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256),
            Dense2DBasicBlock(256, 256),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv1 = sp_tensor.dense()
        x_conv1 = self.conv1(x_conv1)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class DsMiddlePillarEncoder34HA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="DsMiddlePillarEncoder34HA", **kwargs
    ):
        super(DsMiddlePillarEncoder34HA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling(
            mlps=[6 + num_input_features, 16 * double],
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        block = post_act_block

        self.conv1 = nn.Sequential(
            Dense2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg),
        )

        self.conv2 = nn.Sequential(
            block(16 * double, 32 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg),
        )

        self.conv3 = nn.Sequential(
            block(32 * double, 64 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
        )

        self.conv4 = nn.Sequential(
            block(64 * double, 128 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            Dense2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg),
        )

        self.conv5 = nn.Sequential(
            block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv1 = sp_tensor.dense()
        x_conv1 = self.conv1(x_conv1)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class DsMiddlePillarEncoder34x2HA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="DsMiddlePillarEncoder34x2HA", **kwargs
    ):
        super(DsMiddlePillarEncoder34x2HA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling(
            mlps=[6 + num_input_features, 32 * double],
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        block = post_act_block

        # self.conv1 = nn.Sequential(
        #     Dense2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        #     Dense2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        #     Dense2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        # )

        self.conv2 = nn.Sequential(
            # block(16 * double, 32 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg),
        )

        self.conv3 = nn.Sequential(
            block(32 * double, 64 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
        )

        self.conv4 = nn.Sequential(
            block(64 * double, 128 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            Dense2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg),
        )

        self.conv5 = nn.Sequential(
            block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv2 = sp_tensor.dense()
        x_conv2 = self.conv2(x_conv2)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class DsMiddlePillarEncoder34x4HA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="DsMiddlePillarEncoder34x4HA", **kwargs
    ):
        super(DsMiddlePillarEncoder34x4HA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling(
            mlps=[6 + num_input_features, 64 * double],
            bev_size=pillar_cfg['pool3']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        block = post_act_block

        # self.conv1 = nn.Sequential(
        #     Dense2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        #     Dense2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        #     Dense2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        # )

        # self.conv2 = nn.Sequential(
        #     # block(16 * double, 32 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
        #     Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
        #     Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
        #     Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
        #     Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
        # )

        self.conv3 = nn.Sequential(
            # block(32 * double, 64 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
        )

        self.conv4 = nn.Sequential(
            block(64 * double, 128 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            Dense2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg),
        )

        self.conv5 = nn.Sequential(
            block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        # x_conv1 = self.conv1(sp_tensor)
        # x_conv2 = self.conv2(x_conv1)
        x_conv3 = sp_tensor.dense()
        x_conv3 = self.conv3(x_conv3)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )

@BACKBONES.register_module
class DsMiddlePillarEncoder34x8HA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="DsMiddlePillarEncoder34x8HA", **kwargs
    ):
        super(DsMiddlePillarEncoder34x8HA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling(
            mlps=[6 + num_input_features, 128 * double],
            bev_size=pillar_cfg['pool4']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        block = post_act_block

        # self.conv1 = nn.Sequential(
        #     Dense2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        #     Dense2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        #     Dense2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        # )

        # self.conv2 = nn.Sequential(
        #     # block(16 * double, 32 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
        #     Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
        #     Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
        #     Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
        #     Dense2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
        # )
        #
        # self.conv3 = nn.Sequential(
        #     block(32 * double, 64 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
        #     Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
        #     Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
        #     Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
        #     Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
        #     Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
        #     Dense2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg),
        # )

        self.conv4 = nn.Sequential(
            # block(64 * double, 128 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            Dense2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg),
            Dense2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg),
        )

        self.conv5 = nn.Sequential(
            block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        # x_conv1 = self.conv1(sp_tensor)
        # x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(sp_tensor)
        x_conv4 = sp_tensor.dense()
        x_conv4 = self.conv4(x_conv4)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class DsMiddlePillarEncoder2xHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="DsMiddlePillarEncoder2xHA", **kwargs
    ):
        super(DsMiddlePillarEncoder2xHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling(
            mlps=[6 + num_input_features, 32 * double],
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        block = post_act_block

        # self.conv1 = nn.Sequential(
        #     Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        #     Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        # )

        self.conv2 = nn.Sequential(
            Dense2DBasicBlockV(32 * double, 32 * double),
            Dense2DBasicBlock(32 * double, 32 * double),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32 * double, 64 * double, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 64 * double)[1],
            nn.ReLU(),
            Dense2DBasicBlock(64 * double, 64 * double),
            Dense2DBasicBlock(64 * double, 64 * double),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64 * double, 128 * double, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 128 * double)[1],
            nn.ReLU(),
            Dense2DBasicBlock(128 * double, 128 * double),
            Dense2DBasicBlock(128 * double, 128 * double),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256),
            Dense2DBasicBlock(256, 256),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv2 = sp_tensor.dense()
        x_conv2 = self.conv2(x_conv2)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class DsMiddlePillarEncoder4xHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="DsMiddlePillarEncoder4xHA", **kwargs
    ):
        super(DsMiddlePillarEncoder4xHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling(
            mlps=[6 + num_input_features, 64 * double],
            bev_size=pillar_cfg['pool3']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        block = post_act_block

        # self.conv1 = nn.Sequential(
        #     Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        #     Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        # )

        # self.conv2 = nn.Sequential(
        #     Dense2DBasicBlockV(32 * double, 32 * double),
        #     Dense2DBasicBlock(32 * double, 32 * double),
        # )

        self.conv3 = nn.Sequential(
            # nn.Conv2d(32 * double, 64 * double, 3, 2, padding=1, bias=False),
            # build_norm_layer(norm_cfg, 64 * double)[1],
            # nn.ReLU(),
            Dense2DBasicBlockV(64 * double, 64 * double),
            Dense2DBasicBlock(64 * double, 64 * double),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64 * double, 128 * double, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 128 * double)[1],
            nn.ReLU(),
            Dense2DBasicBlock(128 * double, 128 * double),
            Dense2DBasicBlock(128 * double, 128 * double),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256),
            Dense2DBasicBlock(256, 256),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv2 = sp_tensor.dense()
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class DsMiddlePillarEncoder8xHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="DsMiddlePillarEncoder8xHA", **kwargs
    ):
        super(DsMiddlePillarEncoder8xHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling(
            mlps=[6 + num_input_features, 128 * double],
            bev_size=pillar_cfg['pool4']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        block = post_act_block

        # self.conv1 = nn.Sequential(
        #     Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        #     Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        # )

        # self.conv2 = nn.Sequential(
        #     Dense2DBasicBlockV(32 * double, 32 * double),
        #     Dense2DBasicBlock(32 * double, 32 * double),
        # )
        #
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(32 * double, 64 * double, 3, 2, padding=1, bias=False),
        #     build_norm_layer(norm_cfg, 64 * double)[1],
        #     nn.ReLU(),
        #     Dense2DBasicBlock(64 * double, 64 * double),
        #     Dense2DBasicBlock(64 * double, 64 * double),
        # )

        self.conv4 = nn.Sequential(
            # nn.Conv2d(64 * double, 128 * double, 3, 2, padding=1, bias=False),
            # build_norm_layer(norm_cfg, 128 * double)[1],
            # nn.ReLU(),
            Dense2DBasicBlockV(128 * double, 128 * double),
            Dense2DBasicBlock(128 * double, 128 * double),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256),
            Dense2DBasicBlock(256, 256),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv4 = sp_tensor.dense()
        x_conv4 = self.conv4(x_conv4)
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )
