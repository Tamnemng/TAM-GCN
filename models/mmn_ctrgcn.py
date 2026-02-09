import torch
import torch.nn as nn
from .resnet import resnet18 as ResNet
import numpy as np
import sys


class MMNet_CTRGCN(nn.Module):
    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()
        self.resnet = ResNet(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_class) 
        from .ctrgcn import Model as CTRGCN
        self.ctrgcn = CTRGCN(
            num_class=num_class,
            num_point=20, # Ví dụ UCLA là 20 khớp
            num_person=1,
            graph_args=graph_args,
            in_channels=3
        )
        self.temporal_positions = 15
        self.temporal_rgb_frames = 5

    def forward(self, x_, x_rgb):
        with torch.no_grad():
             _, feature = self.ctrgcn.extract_feature(x_)
        intensity_s = (feature*feature).sum(dim=1)**0.5
        intensity_s = intensity_s.cpu().detach().numpy()
        
        feature_s = np.abs(intensity_s)
        if (feature_s.max() - feature_s.min()) != 0:
            feature_s = 255 * (feature_s-feature_s.min()) / (feature_s.max()-feature_s.min())
        
        N, C, T, V, M = x_.size()
        weight = np.full((N, 1, 225, 45*self.temporal_rgb_frames), 0.0) 
        target_joints = [3, 11, 7, 18, 14]
        
        for n in range(N):
            person_idx = 0
            if M > 1:
                if feature_s[n, :, :, 0].mean() < feature_s[n, :, :, 1].mean():
                    person_idx = 1
            
            for j, v in enumerate(target_joints):
                if v < V:
                    feature_val = feature_s[n, :, v, person_idx]
                    temp = np.partition(-feature_val, min(self.temporal_positions, len(feature_val)-1))
                    avg_feat = -temp[:self.temporal_positions].mean()
                    weight[n, 0, 45*j:45*(j+1), :] = avg_feat
        weight_cuda = torch.from_numpy(weight).float().cuda()
        weight_cuda = weight_cuda / 127.0
        rgb_weighted = x_rgb * weight_cuda
        out = self.resnet(rgb_weighted)

        return out