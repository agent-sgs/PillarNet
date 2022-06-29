import torch, math
from typing import List
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
from . import points_cuda
from .group_utils import gather_feature, flatten_indices


@torch.no_grad()
def generate_pillar_indices(bev_size, point_cloud_range, point_batch_cnt, points):
    pillars, pillar_bev_indices = gen_pillar_indices(points, point_batch_cnt, bev_size, point_cloud_range)
    return pillars, pillar_bev_indices


def bev_spatial_shape(point_cloud_range, bev_size):
    W = round((point_cloud_range[3] - point_cloud_range[0]) / bev_size)
    H = round((point_cloud_range[4] - point_cloud_range[1]) / bev_size)
    return int(H), int(W)


@torch.no_grad()
def relative_to_absl(points, pc_range):
    relative = points.clone()
    relative[..., 0] += pc_range[0]
    relative[..., 1] += pc_range[1]
    relative[..., 2] += pc_range[2]
    return relative


@torch.no_grad()
def absl_to_relative(points, pc_range):
    absl = points.clone()
    absl[..., 0] -= pc_range[0]
    absl[..., 1] -= pc_range[1]
    absl[..., 2] -= pc_range[2]
    return absl


class PillarQueryAndGroup(nn.Module):
    def __init__(self, bev_size, point_cloud_range):
        super().__init__()

        self.bev_size = bev_size
        self.spatial_shape = bev_spatial_shape(point_cloud_range, bev_size)
        self.z_center = (point_cloud_range[5] - point_cloud_range[2]) / 2
        self.point_cloud_range = point_cloud_range

    def forward(self, xyz, xyz_batch_cnt, point_features):
        """ batch-wise operation
        Args:
            xyz: (N1+N2..., 3)  relative to the point cloud range
            xyz_batch_cnt: (N1+N2...)
            point_features: (N1+N2..., C)
        Return:
            pillar_indices: indices for resulting sparse pillar tensor
            group_features: (L1+L2..., C)
        """
        pillars, pillar_centers, indice_pairs = gen_indice_pairs(xyz, xyz_batch_cnt,
                                                                 self.bev_size, self.spatial_shape, self.z_center)

        point_set_indices, pillar_set_indices = flatten_indices(indice_pairs)
        group_point_features = gather_feature(point_features, point_set_indices)  # (L, C)
        group_point_xyz = gather_feature(xyz, point_set_indices)  # (L, 3) [xyz]

        group_pillar_centers = gather_feature(pillar_centers, pillar_set_indices)  # (L, 3)  [xyz]
        group_pillar_centers = group_point_xyz - group_pillar_centers

        # group_point_xyz = relative_to_absl(group_point_xyz, self.point_cloud_range)

        # group_features = torch.cat([group_point_features, group_point_xyz - group_pillar_centers], dim=1)
        group_features = torch.cat([group_point_features, group_pillar_centers.detach()], dim=1)

        return pillars, pillar_set_indices, group_features


class GenPillarsIndices(Function):
    @staticmethod
    def forward(ctx, xyz:torch.Tensor, xyz_batch_cnt:torch.Tensor, bev_size, spatial_shape):
        B = xyz_batch_cnt.numel()
        H, W = spatial_shape

        device = xyz.device
        pillar_mask = torch.zeros([B, H, W], dtype=torch.bool, device=device)

        points_cuda.create_pillar_indices_stack_wrapper(bev_size, xyz, xyz_batch_cnt, pillar_mask)

        location = torch.cumsum(pillar_mask.view(-1), 0).int()
        M = location[-1].item()
        pillar_bev_indices = location.view(B, H, W) * pillar_mask - 1
        # create indices (M, 3) [byx]
        pillars = torch.zeros([M, 3], dtype=torch.int32, device=device)
        points_cuda.create_pillar_indices_wrapper(pillar_bev_indices, pillars)

        return pillars, pillar_bev_indices

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


gen_pillar_indices = GenPillarsIndices.apply


class GenIndicePairs(Function):
    @staticmethod
    def forward(ctx, xyz:torch.Tensor, xyz_batch_cnt:torch.Tensor, radius, bev_size, bev_flag, spatial_shape, z_center):
        B = xyz_batch_cnt.numel()
        H, W = spatial_shape
        K = int(round(2 * radius / bev_size)) ** 2

        device = xyz.device
        indice_pairs = torch.full([xyz.shape[0], K], -1, dtype=torch.int32, device=device)
        bev_mask = torch.zeros([B, H, W], dtype=torch.bool, device=device)

        points_cuda.create_pillar_indice_pairs_stack_wrapper(radius, bev_size, xyz, xyz_batch_cnt,
                                                             bev_mask, indice_pairs)
        location = torch.cumsum(bev_mask.view(-1), 0).int()
        M = location[-1].item()
        location = location.view(B, H, W) * bev_mask - 1
        # create indices (M, 3) [byx]
        pillars = torch.zeros([M, 3], dtype=torch.int32, device=device)

        points_cuda.create_pillar_indices_wrapper(location, pillars)
        points_cuda.update_indice_pairs_wrapper(location, indice_pairs)

        # create pillar center [x y z]
        pillar_centers = torch.zeros([M, 3], dtype=torch.float32, device=device, requires_grad=False)
        pillar_centers[:, 0] = (pillars[:, 2] + 0.5) * bev_size
        pillar_centers[:, 1] = (pillars[:, 1] + 0.5) * bev_size
        pillar_centers[:, 2] = z_center

        if bev_flag:
            indice2bev = torch.zeros([M], dtype=torch.int32, device=device)
            points_cuda.create_indice2bev_kernel_wrapper(location, indice2bev)
            return pillar_centers, pillars, indice_pairs, indice2bev
        else:
            return pillar_centers, pillars, indice_pairs, torch.ones(1)

    @staticmethod
    def backward(ctx, a=None, b=None, c=None):
        return None, None

gen_indice_pairs = GenIndicePairs.apply


class GenIndicePairs(Function):
    @staticmethod
    def forward(ctx, xyz:torch.Tensor, xyz_batch_cnt:torch.Tensor, bev_size, spatial_shape, z_center):
        """
        Args:
            xyz: (N1+N2..., 3+C)
            xyz_batch_cnt: (N1, N2, ...)

        Returns:
            pillars: (M1+M2..., 3) [byx]
            pillar_bev_indices: (B, H, W) none(-1)
            pillar_centers: by using pillars yx to calculate centers
            indice_pairs: (N1+N2..., K) neighboring pillars for each point
        """
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()
        assert xyz.shape[1] == 3

        B = xyz_batch_cnt.numel()
        H, W = spatial_shape

        device = xyz.device
        pillar_mask = torch.zeros([B, H, W], dtype=torch.bool, device=device)
        points_cuda.create_pillar_indices_stack_wrapper(bev_size, xyz, xyz_batch_cnt, pillar_mask)
        location = torch.cumsum(pillar_mask.view(-1), 0).int()
        M = location[-1].item()
        pillar_bev_indices = location.view(B, H, W) * pillar_mask - 1
        # create indices (M, 3) [byx]
        pillars = torch.zeros([M, 3], dtype=torch.int32, device=device)
        points_cuda.create_pillar_indices_wrapper(pillar_bev_indices, pillars)

        indice_pairs = torch.full([xyz.shape[0], 1], -1, dtype=torch.int32, device=device)

        # create pillar center [x y z]
        pillar_centers = torch.zeros([pillars.shape[0], 3], dtype=torch.float32, device=device, requires_grad=False)
        pillar_centers[:, 0] = (pillars[:, 2] + 0.5) * bev_size
        pillar_centers[:, 1] = (pillars[:, 1] + 0.5) * bev_size
        pillar_centers[:, 2] = z_center

        points_cuda.create_pillar_indice_pairs_stack_wrapper(bev_size, xyz, xyz_batch_cnt,
                                                                pillar_bev_indices, indice_pairs)

        return pillars, pillar_centers, indice_pairs

    @staticmethod
    def backward(ctx, a=None, b=None, c=None):
        return None, None

gen_indice_pairs = GenIndicePairs.apply
