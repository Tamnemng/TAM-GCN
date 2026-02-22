import torch
import torch.nn as nn
from models.ctrgcn import Model as CTRGCN 
from models.resnet import resnet50

class ResNet_GCN_Attention(nn.Module):
    def __init__(self, num_class=10, num_point=20, num_person=1, graph=None, graph_args=dict(), in_channels_gcn=3, in_channels_rgb=15, drop_out=0, adaptive=True, freeze_gcn=True):
        super(ResNet_GCN_Attention, self).__init__()
        
        # 1. Initialize CTR-GCN
        if graph is None:
            raise ValueError()
        self.gcn = CTRGCN(
            num_class=num_class, 
            num_point=num_point, 
            num_person=num_person, 
            graph=graph, 
            in_channels=in_channels_gcn, 
            drop_out=drop_out, 
            adaptive=adaptive
        )
        
        # Optionally freeze GCN to act solely as a stable feature extractor
        if freeze_gcn:
            for param in self.gcn.parameters():
                param.requires_grad = False
                
        # GCN gives [N, 256, T, V] at the end, so pooled feature depth is 256.
        gcn_feature_dim = 256 
        
        # 2. Initialize ResNet50
        self.resnet = resnet50(pretrained=True)
        # ResNet50 layer4 output is [N, 2048, 7, 7]
        resnet_feature_dim = 2048
        
        # Inflate conv1 to handle in_channels_rgb (e.g., 15 for 5 frames) instead of 3
        if in_channels_rgb != 3:
            original_conv1 = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(
                in_channels_rgb,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False
            )
            nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
            # Initialize with averaged pretrained weights
            with torch.no_grad():
                # repeat the weights along the channel dimension and scale
                # original weight shape: [64, 3, 7, 7] -> new shape: [64, 15, 7, 7]
                self.resnet.conv1.weight[:] = original_conv1.weight.repeat(1, in_channels_rgb // 3, 1, 1) / (in_channels_rgb // 3)
        
        # Remove original ResNet FC and pooling (we'll do it manually after attention)
        self.resnet.fc = nn.Identity()
        self.resnet.avgpool = nn.Identity()
        
        # 3. Cross-Modal Attention Module
        # We want to transform the 256-d GCN feature into a 2048-d channel attention vector
        self.attention_transform = nn.Sequential(
            nn.Linear(gcn_feature_dim, resnet_feature_dim // 2),
            nn.BatchNorm1d(resnet_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(resnet_feature_dim // 2, resnet_feature_dim),
            nn.Sigmoid() # Scale values between 0 and 1
        )
        
        # 4. Final Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(resnet_feature_dim, num_class)
        
    def forward(self, x_gcn, x_rgb):
        """
        x_gcn: [N, C, T, V, M] (Skeleton data)
        x_rgb: [N, 3, 224, 224] (RGB image data)
        """
        # ==================================
        # 1. Extract GCN Semantic Guidance
        # ==================================
        # .extract_feature() returns the tensor right before FC layer.
        # Shape output from CTRGCN extract_feature: [N, 256, 13, 20, 1]
        f_gcn, _ = self.gcn.extract_feature(x_gcn) 
        
        # Pool across Time (T), Joints (V), and Person (M) to get a single vector per video
        f_gcn = f_gcn.mean(dim=(2, 3, 4)) # Shape: [N, 256]
        
        # Generate Attention Weights
        # Shape: [N, 2048]
        att_weights = self.attention_transform(f_gcn) 
        # Reshape for broadcasting over the spatial grid [N, 2048, 1, 1]
        att_weights = att_weights.unsqueeze(-1).unsqueeze(-1) 
        
        # ==================================
        # 2. Extract ResNet Spatial Maps
        # ==================================
        # Forward pass through resnet up to layer4
        f_rgb = self.resnet.conv1(x_rgb)
        f_rgb = self.resnet.bn1(f_rgb)
        f_rgb = self.resnet.relu(f_rgb)
        f_rgb = self.resnet.maxpool(f_rgb)

        f_rgb = self.resnet.layer1(f_rgb)
        f_rgb = self.resnet.layer2(f_rgb)
        f_rgb = self.resnet.layer3(f_rgb)
        f_rgb = self.resnet.layer4(f_rgb) 
        # Shape: [N, 2048, 7, 7]
        
        # ==================================
        # 3. Apply Cross-Modal Attention
        # ==================================
        # Multiply ResNet feature maps by GCN-derived attention weights
        f_attended = f_rgb * att_weights 
        
        # ==================================
        # 4. Classification
        # ==================================
        out = self.global_pool(f_attended) # Shape: [N, 2048, 1, 1]
        out = torch.flatten(out, 1)        # Shape: [N, 2048]
        
        out = self.classifier(out)         # Shape: [N, num_class]
        
        return out
