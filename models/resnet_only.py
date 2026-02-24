import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class Model(nn.Module):
    def __init__(self, num_class=10, pretrained=True, **kwargs):
        super(Model, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_class)

        self.stgcn = ''
        self.temporal_positions = 15
        self.temporal_rgb_frames = 5

    def forward(self, x_, x_rgb):
        predict, feature = self.stgcn.extract_feature(x_)
        intensity_s = (feature*feature).sum(dim=1)**0.5

        intensity_s = intensity_s.cpu().detach().numpy()
        feature_s = np.abs(intensity_s)
        feature_s = 255 * (feature_s-feature_s.min()) / (feature_s.max()-feature_s.min())
        N, C, T, V, M = x_.size()

        weight = np.full((N, 1, 225, 45*self.temporal_rgb_frames),0)
        for n in range(N):
            if True:#feature_s[n, :, :, 0].mean(1).mean(0) > feature_s[n, :, :, 1].mean(1).mean(0):
                for j, v in enumerate([3, 11, 7, 18, 14]):
                    # use TOP 10 values along the temporal dimension
                    feature = feature_s[n, :, v, 0]
                    kth = min(self.temporal_positions, feature.shape[0] - 1)
                    temp = np.partition(-feature, kth)
                    feature = -temp[:kth].mean()
                    weight[n, 0, 45*j:45*(j+1), :] = feature[np.newaxis, np.newaxis]
            else:
                for j, v in enumerate([3, 11, 7, 18, 14]):
                    # use TOP 10 values along the temporal dimension
                    feature = feature_s[n, :, v, 1]
                    kth = min(self.temporal_positions, feature.shape[0] - 1)
                    temp = np.partition(-feature, kth)
                    feature = -temp[:kth].mean()
                    weight[n, 0, 45*j:45*(j+1), :] = feature[np.newaxis, np.newaxis]

        weight_cuda = torch.from_numpy(weight).float().cuda()
        weight_cuda = weight_cuda / 127
        rgb_weighted = x_rgb * weight_cuda

        out = self.model(rgb_weighted)

        return out