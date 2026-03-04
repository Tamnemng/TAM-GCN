"""V2: V0 + Average Joint Grouping.

Only change from V0: instead of picking 1 representative joint per body part,
we average ALL joints in each body-part group:
  Torso [0,1,2,3], L-Arm [4,5,6,7], R-Arm [8,9,10,11],
  L-Leg [12,13,14,15], R-Leg [16,17,18,19]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CrossModalAttention(nn.Module):
    """Cross-Modal Attention: injects skeleton features into ResNet feature maps.
    
    Identical to V0 — CBAM-style channel + spatial attention.
    """
    def __init__(self, rgb_channels, skel_grid_size=25, reduction=4):
        super().__init__()
        
        # 1. CHANNEL ATTENTION
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False),
            nn.Sigmoid()
        )
        
        # 2. SPATIAL ATTENTION (same as V0: Conv2d(3,1,7x7)+BN+Sigmoid)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, rgb_feat, skel_grid, exp_type='normal'):
        B, C, H, W = rgb_feat.shape
        
        if exp_type == 'noise':
            skel_grid = torch.randn_like(skel_grid)
        elif exp_type == 'ones':
            skel_grid = torch.ones_like(skel_grid)
        elif exp_type == 'zeros':
            skel_grid = torch.zeros_like(skel_grid)
        
        # --- Channel Attention ---
        skel_flat = skel_grid.view(B, -1)
        ch_attn = self.channel_attn(skel_flat).unsqueeze(-1).unsqueeze(-1)
        feat_ca = rgb_feat * ch_attn
        
        if exp_type == 'no_spatial':
            return rgb_feat + feat_ca
            
        # --- Spatial Attention (cross-modal) ---
        skel_sp = F.interpolate(skel_grid, size=(H, W), mode='bilinear', align_corners=False)
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)
        sp_input = torch.cat([rgb_max, rgb_avg, skel_sp], dim=1)
        sp_attn = self.spatial_conv(sp_input)
        
        modulated = feat_ca * sp_attn
        return rgb_feat + modulated


class Model(nn.Module):
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5, exp_type='normal'):
        super(Model, self).__init__()
        
        self.exp_type = exp_type
        self.ctrgcn = ''
        self.temporal_rgb_frames = temporal_rgb_frames
        
        # ResNet-50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(resnet.fc.in_features, num_class)
        
        # Cross-modal attention at layer2 output (512 channels)
        self.cross_attn = CrossModalAttention(
            rgb_channels=512,
            skel_grid_size=5 * temporal_rgb_frames,
            reduction=4
        )
        
        # ★ V2 CHANGE: Average joint groups instead of single representative joints
        self.part_groups = [
            [0, 1, 2, 3],      # Torso
            [4, 5, 6, 7],      # Left Arm
            [8, 9, 10, 11],    # Right Arm
            [12, 13, 14, 15],  # Left Leg
            [16, 17, 18, 19],  # Right Leg
        ]
        
    def _build_skel_grid(self, intensity_norm):
        """Build (B, 1, 5, T_frames) spatiotemporal skeleton feature grid.
        
        V2: Each body part value = average of ALL joints in the group.
        """
        B, T_new, V, M = intensity_norm.shape
        T_frames = self.temporal_rgb_frames
        
        part_features = []
        for group in self.part_groups:
            # Average all joints in this body-part group
            joint_feats = torch.stack(
                [intensity_norm[:, :, v, 0] for v in group], dim=-1
            )  # (B, T_new, len(group))
            part_feat = joint_feats.mean(dim=-1)  # (B, T_new)
            part_features.append(part_feat)
        part_features = torch.stack(part_features, dim=1)  # (B, 5, T_new)
        
        # Adaptive pool T_new → T_frames temporal bins
        part_features = F.adaptive_avg_pool1d(part_features, T_frames)
        
        # Normalize (same as V0)
        part_features = part_features / 127.0
        
        return part_features.unsqueeze(1)  # (B, 1, 5, T_frames)
        
    def forward(self, x_s, x_rgb):
        # x_s: (B, C, T, V, M) = (B, 3, 52, 20, 1)
        # x_rgb: (B, 3, 224, 224)
        
        # ===== 1. Skeleton branch =====
        with torch.no_grad():
            _, feature_s = self.ctrgcn.extract_feature(x_s)
            
            intensity = (feature_s * feature_s).sum(dim=1) ** 0.5
            intensity = torch.abs(intensity)
            
            B = intensity.shape[0]
            flat = intensity.view(B, -1)
            f_min = flat.min(dim=1, keepdim=True)[0]
            f_max = flat.max(dim=1, keepdim=True)[0]
            diff = f_max - f_min
            diff[diff == 0] = 1e-6
            flat_norm = 255.0 * (flat - f_min) / diff
            intensity_norm = flat_norm.view_as(intensity)
        
        skel_grid = self._build_skel_grid(intensity_norm.detach())
        
        # ===== 2. RGB branch with cross-modal injection =====
        x = self.stem(x_rgb)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.cross_attn(x, skel_grid, exp_type=self.exp_type)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.fc(x)
        
        return output
