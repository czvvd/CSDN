import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import math 
import os
from torch.autograd import Variable

def gen_grid_up(up_rtatio):
    sqrted = int(math.sqrt(up_rtatio))+1
    for i in range(1,sqrted+1).__reversed__():
        if (up_rtatio%i) == 0:
            num_x = i
            num_y = up_rtatio//i
            break
    grid_x = torch.linspace(-0.2,0.2, num_x)
    grid_y = torch.linspace(-0.2,0.2, num_y)

    x, y = torch.meshgrid(grid_x,grid_y)
    grid = torch.reshape(torch.stack([x,y], axis=-1), [-1,2])
    return grid.to('cuda')

class ProjectionLayer(nn.Module):
    def __init__(self):
        super(ProjectionLayer, self).__init__()

    def forward(self, img_features, input, batch):

        self.img_feats = img_features

        h = 248 * torch.div(input[:, 1], input[:, 2]) + 111.5
        w = 248 * torch.div(input[:, 0], -input[:, 2]) + 111.5

        h = torch.clamp(h, min = 0, max = 223)
        w = torch.clamp(w, min = 0, max = 223)

        img_sizes = [56, 28, 14, 7]
        out_dims = [64, 128, 256, 512]
        feats = []

        for i in range(4):
            out = self.project(i, h, w, img_sizes[i], out_dims[i],batch)
            feats.append(out)
            
        output = torch.cat(feats, 1)
        
        return output

    def project(self, index, h, w, img_size, out_dim,batch):

        img_feat = self.img_feats[index][batch]
        x = h / (224. / img_size)
        y = w / (224. / img_size)

        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()

        x2 = torch.clamp(x2, max = img_size - 1)
        y2 = torch.clamp(y2, max = img_size - 1)

        Q11 = img_feat[:, x1, y1].clone()
        Q12 = img_feat[:, x1, y2].clone()
        Q21 = img_feat[:, x2, y1].clone()
        Q22 = img_feat[:, x2, y2].clone()

        x, y = x.long(), y.long()

        weights = torch.mul(x2 - x, y2 - y)
        Q11 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q11, 0, 1))

        weights = torch.mul(x2 - x, y - y1)
        Q12 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q12, 0 ,1))

        weights = torch.mul(x - x1, y2 - y)
        Q21 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q21, 0, 1))

        weights = torch.mul(x - x1, y - y1)
        Q22 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q22, 0, 1))

        output = Q11 + Q21 + Q12 + Q22

        return output

class PointNetRes(nn.Module):
    def __init__(self):
        super(PointNetRes, self).__init__()
        self.projection = ProjectionLayer()
        self.conv1 = torch.nn.Conv1d(3011, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 32, 1)
        self.conv6 = torch.nn.Conv1d(32, 3, 1)


        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(32)
        self.th = nn.Tanh()
        self.fc = nn.Linear(2048,2048)

    def forward(self, input):
        level0 = input[0]
        code = input[1]
        global_code = input[2]
        img_fea = input[3]
        batch_size = level0.shape[0]
        npoints = level0.size()[2]
        img_proj_feat = []
        for i in range(batch_size):
            img_proj_feat.append(torch.unsqueeze(self.projection(img_fea, level0[i].permute(1, 0), i), 0))
        img_proj_feat = torch.cat(img_proj_feat, dim=0)
        global_feat = code.unsqueeze(2).repeat(1, 1, 2048)
        generate_feat = global_code.unsqueeze(2).repeat(1, 1, 2048)

        img_proj_feat = img_proj_feat.permute(0, 2, 1)
        img_proj_feat = self.fc(img_proj_feat)
        x = torch.cat([level0, global_feat, generate_feat, img_proj_feat], axis=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.th(self.conv6(x))
        return x.permute(0,2,1)

class PartRefinement(nn.Module):
    def __init__(self,step_ratio = 2, up_ratio = 2):
        super(PartRefinement,self).__init__()
        self.step_ratio = step_ratio
        self.up_ratio = up_ratio
        self.projection = ProjectionLayer()


        self.mlp1 = nn.Linear(1024,128)
        self.conv1d_1 = nn.Conv1d(3013,1024,1)
        self.conv1d_2 = nn.Conv1d(1024,128,1)
        self.conv1d_3 = nn.Conv1d(128,64,1)

        self.conv2d_1 = nn.Conv2d(64,64,[1,self.up_ratio])
        self.conv2d_2 = nn.Conv2d(64,128,[1,1])
        self.conv2d_3 = nn.Conv2d(64,32,[1,1])

        self.conv1d_4 = nn.Conv1d(64,512,1)
        self.conv1d_5 = nn.Conv1d(512,512,1)
        self.conv1d_6 = nn.Conv1d(512,6,1)

        self.fc = nn.Linear(1*1024,1*1024)
        self.feat = None


    def forward(self,x, rate):
        # x = [concat, partial point feat]
        # concat and downsample point clouds should be 1024
        level0 = x[0]
        code = x[1]
        global_code = x[2]
        img_fea = x[3]
        batch_size = level0.shape[0]
        input_point_nums = level0.shape[2]
        # for i,key in enumerate(img_fea):
        #     img_fea[i] = torch.squeeze(key)
        # level0_squeeze = torch.squeeze(level0)
        img_proj_feat = []
        for i in range(batch_size):
            img_proj_feat.append(torch.unsqueeze(self.projection(img_fea,level0[i].permute(1,0),i),0))
        img_proj_feat = torch.cat(img_proj_feat,dim=0)
        # img_proj_feat = self.projection(img_fea,level0_squeeze.permute(1,0))

        num_fine = rate*input_point_nums
        grid = gen_grid_up(rate**(0+1))
        grid = grid.unsqueeze(0).permute(0,2,1)
        grid_feat = grid.repeat(level0.shape[0],1,int(input_point_nums/2))

        point_out = level0.unsqueeze(2).repeat(1,1,rate,1)
        point_out =  torch.reshape(point_out,[-1,3,num_fine])

        point_feat = level0.unsqueeze(2).repeat(1,1,1,1)
        point_feat = torch.reshape(point_feat,[-1,3,int(num_fine/2)])

        global_feat = code.unsqueeze(2).repeat(1,1,int(num_fine/2))
        generate_feat = global_code.unsqueeze(2).repeat(1,1,int(num_fine/2))

        img_proj_feat = img_proj_feat.permute(0,2,1)
        img_proj_feat = self.fc(img_proj_feat)
        
        feat = torch.cat([grid_feat,point_feat,global_feat,generate_feat,img_proj_feat],axis=1)
        self.feat = feat

        # Dynamic Offset Predictor
        feat1 = self.conv1d_1(feat)
        feat1 = self.conv1d_2(feat1)
        feat1 = self.conv1d_3(feat1)
        feat1 = F.relu(feat1)

        feat2 = feat1.unsqueeze(-1).repeat(1,1,1,2)
        feat2 = self.conv2d_1(feat2)
        feat2 = self.conv2d_2(feat2)

        feat2 = feat2.view(feat2.shape[0], self.up_ratio, 64, -1).permute(0, 2, 1, 3)
        feat2 = self.conv2d_3(feat2)
        feat2 = feat2.view(feat2.shape[0], 64, -1)

        feat = feat1 + feat2

        feat = self.conv1d_4(feat)
        feat = self.conv1d_5(feat)
        feat = self.conv1d_6(feat)
        offset = feat.view(-1,3,2048)

        fine = offset + point_out

        return offset.permute(0,2,1)

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

class partial_refine(nn.Module):
    def __init__(self,k,dim = [32,128,512]):
        super(partial_refine, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(dim[0])
        self.bn2 = nn.BatchNorm2d(dim[1])
        self.bn3 = nn.BatchNorm2d(dim[2])
        self.conv1 = nn.Sequential(nn.Conv2d(6, dim[0], kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(dim[0],dim[1], kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(dim[1], dim[2], kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self,partial,coarse):
        dist_self = square_distance(coarse,coarse)
        knn_self_idx = dist_self.argsort()[:, :, :self.k]
        knn_self_xyz = index_points(coarse, knn_self_idx)
        dists = square_distance(coarse,partial)
        knn_idx = dists.argsort()[:, :, :self.k]
        knn_xyz = index_points(partial,knn_idx)

        knn_cat = torch.cat([knn_xyz,knn_self_xyz],2)                    # (b,n,2k,3)
        coarse_dupk = torch.unsqueeze(coarse,2).repeat(1,1,2*self.k,1)   # (b,n,2k,3)
        knn_cat = knn_cat - coarse_dupk
        knn_xyz = torch.cat([coarse_dupk,knn_cat],3).permute(0, 3, 1, 2).contiguous()  # (b,6,n,2k)
        # knn_self_xyz = knn_self_xyz - coarse_dupk
        # knn_xyz = knn_xyz - coarse_dupk
        # knn_xyz = torch.cat([coarse_dupk,knn_self_xyz,knn_xyz],3) .permute(0, 3, 1, 2).contiguous()
        offset = self.conv1(knn_xyz)                                # (b,64,n,2k)
        offset = self.conv2(offset)
        offset = self.conv3(offset).max(dim=-1, keepdim=False)[0]

        return offset

class image_refine(nn.Module):
    def __init__(self,dim = [1024,512]):
        super(image_refine, self).__init__()
        self.conv_shortcut = nn.Conv1d(963,dim[1],1)
        self.conv1 = torch.nn.Conv1d(963, dim[0], 1)
        self.conv2 = torch.nn.Conv1d(dim[0], dim[1], 1)
        self.bn1 = torch.nn.BatchNorm1d(dim[0])

    def forward(self,x,img_feat):
        offset = torch.cat([x,img_feat],1)
        shrotcut = self.conv_shortcut(offset)
        offset = F.relu(self.bn1(self.conv1(offset)))
        offset = self.conv2(offset)+shrotcut
        return offset

class DualRefine(nn.Module):
    def __init__(self,k=16):
        super(DualRefine, self).__init__()
        self.projection = ProjectionLayer()
        self.partial_refine = partial_refine(k=k)
        self.image_refine = image_refine()
        self.conv1 = torch.nn.Conv1d(512, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 32, 1)
        self.conv4 = torch.nn.Conv1d(32, 3, 1)

        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.th = nn.Tanh()

    def forward(self, coarse, partial,img_fea):
        batch_size = coarse.shape[0]
        offset_partial = self.partial_refine(partial.permute(0,2,1),coarse.permute(0,2,1))

        img_proj_feat = []
        for i in range(batch_size):
            img_proj_feat.append(torch.unsqueeze(self.projection(img_fea, coarse[i].permute(1, 0), i), 0))
        img_proj_feat = torch.cat(img_proj_feat, dim=0)
        img_proj_feat = img_proj_feat.permute(0, 2, 1)
        offset_img = self.image_refine(coarse,img_proj_feat)

        x = offset_partial+offset_img

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x.permute(0,2,1)


