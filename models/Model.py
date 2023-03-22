from models.part_refinement import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()
from models.utlis import PointNetFeatureExtractor,fps_subsample

class CSDN(nn.Module):
    def __init__(self,k=16):
        super(CSDN, self).__init__()
        self.Cdecoder = nn.ModuleList([PointGenCon(bottleneck_size = 1026) for i in range(0,4)])
        self.pointnet_encoder = PointNetFeatureExtractor(
            in_channels=3,
            feat_size=1024,
            layer_dims=[64, 64, 64, 128],
            transposed_input=True)

        self.part_refinement = DualRefine(k=k)

        # view encoder
        self.conv0_1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv0_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 224 -> 112
        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 112 -> 56
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 56 -> 28
        self.conv4_1 = nn.Conv2d(128, 256, 5, stride=2, padding=2)  # 28 -> 14
        self.conv5_1 = nn.Conv2d(256, 512, 5, stride=2, padding=2)  # 14 -> 7
        self.viewpool = nn.AvgPool2d((7,7))

        self.mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, get_num_adain_params(self.Cdecoder[0]))
        )

    def forward(self,view,partial):
        b = partial.size(0)
        partial_point_feat = self.pointnet_encoder(partial.permute(0, 2, 1))

        x = F.relu(self.conv0_1(view))
        x = F.relu(self.conv0_2(x))
        x = F.relu(self.conv1_1(x))
        x1 = x  # 112
        x = F.relu(self.conv2_1(x))
        x2 = x  # 56
        x = F.relu(self.conv3_1(x))
        x3 = x  # 28
        x = F.relu(self.conv4_1(x))
        x4 = x  # 14
        x = F.relu(self.conv5_1(x))
        x5 = x  # 7
        style = torch.squeeze(self.viewpool(x5))

        params = self.mlp(style)
        assign_adain_params(params, self.Cdecoder[0])
        assign_adain_params(params, self.Cdecoder[1])
        assign_adain_params(params, self.Cdecoder[2])
        assign_adain_params(params, self.Cdecoder[3])

        outs = []

        for i in range(0, 4):
            rand_grid = torch.cuda.FloatTensor(b, 2, 512)
            rand_grid.data.uniform_(0, 1)
            y = partial_point_feat.unsqueeze(2).expand(b, partial_point_feat.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            y= self.Cdecoder[i](y)
            outs.append(y)
        outs = torch.cat(outs, 2).contiguous()

        reconstructed_pc = outs.permute(0,2,1)
        concat_pc = torch.cat([reconstructed_pc, partial], dim=1)
        coarse_pc = fps_subsample(concat_pc, 2048)

        offset = self.part_refinement(coarse_pc.permute(0, 2, 1), partial.permute(0, 2, 1), [x2, x3, x4, x5])

        fine_point_cloud = coarse_pc + offset

        return fine_point_cloud, reconstructed_pc, coarse_pc

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 8192, output_size = 3):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, output_size, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

        self.adain1 = AdaptiveInstanceNorm1d(self.bottleneck_size)
        self.adain2 = AdaptiveInstanceNorm1d(self.bottleneck_size // 2)
        self.adain3 = AdaptiveInstanceNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        f1 = self.conv1(x)
        f1_af = F.relu(self.bn1(self.adain1(f1)))
        f2 = self.conv2(f1_af)
        f2_af = F.relu(self.bn2(self.adain2(f2)))
        f3 = self.conv3(f2_af)
        f3_af = F.relu(self.bn3(self.adain3(f3)))
        x = self.th(self.conv4(f3_af))
        return x

def assign_adain_params(adain_params, model):

    """
    inputs:
    - adain_params: b x parameter_size
    - model: nn.module

    function:
    assign_adain_params
    """
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            mean = adain_params[:, : m.num_features]
            std = adain_params[:, m.num_features : 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features :]

class AdaptiveInstanceNorm1d(nn.Module):
    """
    input:
    - inp: (b, c, m)

    output:
    - out: (b, c, m')
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped,  # (1,6,512)
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )
        return out.view(b, c, *x.size()[2:])

def get_num_adain_params(model):
    """
    input:
    - model: nn.module

    output:
    - num_adain_params: int
    """
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            num_adain_params += 2 * m.num_features
    return num_adain_params


