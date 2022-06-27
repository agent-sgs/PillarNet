import torch.nn as nn
from typing import List
try:
    import spconv.pytorch as spconv
except:
    import spconv
from .pillar_utils import PillarQueryAndGroup, bev_spatial_shape
from .scatter_utils import scatter_max


class PillarMaxPooling(nn.Module):
    def __init__(self, mlps: List[int], bev_size:float, point_cloud_range:List[float]):
        super().__init__()

        self.bev_width, self.bev_height = bev_spatial_shape(point_cloud_range, bev_size)

        self.groups = PillarQueryAndGroup(bev_size, point_cloud_range)

        shared_mlp = []
        for k in range(len(mlps) - 1):
            shared_mlp.extend([
                # nn.Conv1d(mlps[k], mlps[k + 1], kernel_size=1, bias=False),
                nn.Linear(mlps[k], mlps[k + 1], bias=False),
                nn.BatchNorm1d(mlps[k + 1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ])
        self.shared_mlps = nn.Sequential(*shared_mlp)

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, pt_feature):
        """
        Args:
            xyz: (N1+N2..., 3)
            xyz_batch_cnt:  (N1, N2, ...)
            point_features: (N1+N2..., C)
            spatial_shape: [B, H, W]
        Return:
            pillars: (M1+M2..., 3) [byx]
            pillar_features: (M, C)
        """
        B = xyz_batch_cnt.shape[0]
        pillar_indices, pillar_set_indices, group_features = self.groups(xyz, xyz_batch_cnt, pt_feature)

        # group_features = self.shared_mlps(group_features.transpose(1, 0).unsqueeze(dim=0))  # (1, C, L)
        group_features = self.shared_mlps(group_features)  # (1, C, L)
        group_features = group_features.transpose(1, 0).contiguous()

        pillar_features = scatter_max(group_features, pillar_set_indices, pillar_indices.shape[0])   # (C, M)
        pillar_features = pillar_features.transpose(1, 0)   # (M, C)

        return spconv.SparseConvTensor(pillar_features, pillar_indices, (self.bev_height, self.bev_width), B)

