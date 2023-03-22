import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable
from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation

class PointNetFeatureExtractor(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 feat_size: int = 1024,
                 layer_dims: Iterable[int] = [64, 128],
                 global_feat: bool = True,
                 activation=F.relu,
                 batchnorm: bool = True,
                 transposed_input: bool = False):
        super(PointNetFeatureExtractor, self).__init__()

        if not isinstance(in_channels, int):
            raise TypeError('Argument in_channels expected to be of type int. '
                            'Got {0} instead.'.format(type(in_channels)))
        if not isinstance(feat_size, int):
            raise TypeError('Argument feat_size expected to be of type int. '
                            'Got {0} instead.'.format(type(feat_size)))
        if not hasattr(layer_dims, '__iter__'):
            raise TypeError('Argument layer_dims is not iterable.')
        for idx, layer_dim in enumerate(layer_dims):
            if not isinstance(layer_dim, int):
                raise TypeError('Elements of layer_dims must be of type int. '
                                'Found type {0} at index {1}.'.format(
                                    type(layer_dim), idx))
        if not isinstance(global_feat, bool):
            raise TypeError('Argument global_feat expected to be of type '
                            'bool. Got {0} instead.'.format(
                                type(global_feat)))

        # Store feat_size as a class attribute
        self.feat_size = feat_size

        # Store activation as a class attribute
        self.activation = activation

        # Store global_feat as a class attribute
        self.global_feat = global_feat

        # Add in_channels to the head of layer_dims (the first layer
        # has number of channels equal to `in_channels`). Also, add
        # feat_size to the tail of layer_dims.
        if not isinstance(layer_dims, list):
            layer_dims = list(layer_dims)
        layer_dims.insert(0, in_channels)
        layer_dims.append(feat_size)

        self.conv_layers = nn.ModuleList()
        if batchnorm:
            self.bn_layers = nn.ModuleList()
        for idx in range(len(layer_dims) - 1):
            self.conv_layers.append(nn.Conv1d(layer_dims[idx],
                                              layer_dims[idx + 1], 1))
            if batchnorm:
                self.bn_layers.append(nn.BatchNorm1d(layer_dims[idx + 1]))

        # Store whether or not to use batchnorm as a class attribute
        self.batchnorm = batchnorm

        self.transposed_input = transposed_input

    def forward(self, x: torch.Tensor):
        if not self.transposed_input:
            x = x.transpose(1, 2)

        # Number of points
        num_points = x.shape[2]

        # By default, initialize local features (per-point features)
        # to None.
        local_features = None

        # Apply a sequence of conv-batchnorm-nonlinearity operations

        # For the first layer, store the features, as these will be
        # used to compute local features (if specified).
        if self.batchnorm:
            x = self.activation(self.bn_layers[0](self.conv_layers[0](x)))
        else:
            x = self.activation(self.conv_layers[0](x))
        if self.global_feat is False:
            local_features = x

        # Pass through the remaining layers (until the penultimate layer).
        for idx in range(1, len(self.conv_layers) - 1):
            if self.batchnorm:
                x = self.activation(self.bn_layers[idx](
                    self.conv_layers[idx](x)))
            else:
                x = self.activation(self.conv_layers[idx](x))

        # For the last layer, do not apply nonlinearity.
        if self.batchnorm:
            x = self.bn_layers[-1](self.conv_layers[-1](x))
        else:
            x = self.conv_layers[-1](x)

        # Max pooling.
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.feat_size)

        # If extracting global features, return at this point.
        if self.global_feat:
            return x

        # If extracting local features, compute local features by
        # concatenating global features, and per-point features
        x = x.view(-1, self.feat_size, 1).repeat(1, 1, num_points)
        return torch.cat((x, local_features), dim=1)

def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd
