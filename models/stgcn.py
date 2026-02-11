"""
ST-GCN model adapted for UCLA 20-joint skeleton.
Uses learnable edge_importance weights per layer — these are GLOBAL params
(same for all samples), making them ideal for consistent body-part weighting.

Reference: "Spatial Temporal Graph Convolutional Networks for Skeleton-Based 
Action Recognition" (Yan et al., AAAI 2018)
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class ConvTemporalGraphical(nn.Module):
    """Spatial graph convolution layer.
    
    Performs graph convolution: output = sum_k( conv_k(x) * A_k )
    where A_k is the k-th partition of the adjacency matrix.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A


class st_gcn(nn.Module):
    """One ST-GCN block: GCN + TCN + residual."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class Model(nn.Module):
    """ST-GCN model with learnable edge importance weighting.
    
    Key advantage over CTR-GCN for weight extraction:
    - edge_importance is a GLOBAL parameter (nn.Parameter), same for ALL samples
    - Directly weights edges in the adjacency matrix → interpretable
    - No per-sample variance → consistent body-part importance
    
    Args:
        in_channels: Input channels (3 for xyz coordinates)
        num_class: Number of output classes (4 for group labels)
        num_point: Number of skeleton joints (20 for UCLA)
        num_person: Number of persons per frame (1 for UCLA)
        graph: Graph class path string
        graph_args: Arguments for graph construction
        edge_importance_weighting: Whether to learn edge importance
    """
    def __init__(self, in_channels=3, num_class=4, num_point=20, num_person=1,
                 graph=None, graph_args=dict(), edge_importance_weighting=True,
                 dropout=0, **kwargs):
        super().__init__()

        # Load graph
        if graph is None:
            raise ValueError("Graph class must be specified")
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # Build networks
        spatial_kernel_size = A.size(0)  # 3 for spatial partitioning
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # Edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # Classification head
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        
        if dropout:
            self.drop_out = nn.Dropout(dropout)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        # Handle input format: (N, C, T, V, M) or (N, T, VC)
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        
        N, C, T, V, M = x.size()
        
        # Data normalization
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # Forward through ST-GCN layers
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # Global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # Prediction
        x = self.drop_out(x)
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x

    def extract_feature(self, x):
        """Extract features — same as forward but returns intermediate features."""
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        
        N, C, T, V, M = x.size()
        
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # Prediction branch
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature
    
    def get_edge_importance_per_joint(self):
        """
        Tính importance trung bình cho mỗi joint dựa trên edge_importance.
        
        Edge importance shape per layer: (K, V, V) where K=3 partitions, V=20 joints.
        importance[k, i, j] = weight of edge from joint j → joint i in partition k.
        
        Per-joint importance = trung bình tổng weight của tất cả edges liên quan đến joint đó,
        across tất cả layers.
        
        Returns:
            joint_importance: numpy array shape (V,) — importance score per joint
        """
        V = self.A.size(1)
        joint_scores = np.zeros(V)
        
        for importance in self.edge_importance:
            imp = importance.detach().cpu().numpy()  # (K, V, V)
            # Sum over all partitions: incoming + outgoing edges per joint
            for k in range(imp.shape[0]):
                joint_scores += imp[k].sum(axis=0)  # outgoing: how much joint j sends
                joint_scores += imp[k].sum(axis=1)  # incoming: how much joint i receives
        
        # Normalize
        joint_scores = joint_scores / joint_scores.max()
        return joint_scores
