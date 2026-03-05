"""V5: V0 + Learned Joint-to-Part Aggregation via Conv1d.

Change from V0: thay vì chọn 1 joint đại diện per body part (hardcoded),
dùng Conv1d(V=20, 5, kernel_size=1) để HỌC trọng số mapping
20 joints → 5 body parts trước khi đưa vào skeleton grid.

Giống v2 (dùng cả 20 joints) nhưng thay mean() bằng Conv1d có learnable weights.
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
            skel_grid_size=5 * temporal_rgb_frames,  # 5 parts × T_frames = 25
            reduction=4
        )

        # ★ V5 CHANGE: Conv1d(V=20, 5, kernel_size=1) để học trọng số
        # mapping từ 20 joints → 5 body parts (thay hardcode)
        # Input: (B, V=20, T_new), Output: (B, 5, T_new)
        self.joint_to_part = nn.Sequential(
            nn.Conv1d(20, 5, kernel_size=1, bias=False),
            nn.BatchNorm1d(5),
            nn.ReLU(inplace=True),
        )
        # Init Conv1d weights theo nhóm v2: mỗi part = mean của 4 joints trong nhóm
        # Khởi đầu từ v2 behavior, rồi fine-tune
        part_groups = [
            [0, 1, 2, 3],      # Torso
            [4, 5, 6, 7],      # Left Arm
            [8, 9, 10, 11],    # Right Arm
            [12, 13, 14, 15],  # Left Leg
            [16, 17, 18, 19],  # Right Leg
        ]
        with torch.no_grad():
            self.joint_to_part[0].weight.zero_()  # (5, 20, 1)
            for i, group in enumerate(part_groups):
                for j in group:
                    self.joint_to_part[0].weight[i, j, 0] = 1.0 / len(group)

    def _build_skel_grid(self, intensity_norm):
        """Build (B, 1, 5, T_frames) spatiotemporal skeleton feature grid.

        V5: Dùng Conv1d để học trọng số 20 joints → 5 body parts.
        """
        B, T_new, V, M = intensity_norm.shape
        T_frames = self.temporal_rgb_frames

        # intensity_norm: (B, T_new, V=20, M=1) → (B, V, T_new)
        part_features = intensity_norm[:, :, :, 0]      # (B, T_new, V)
        part_features = part_features.permute(0, 2, 1)  # (B, V=20, T_new)

        # Learned joint → part aggregation
        part_features = self.joint_to_part(part_features)  # (B, 5, T_new)

        # Adaptive pool T_new → T_frames temporal bins
        part_features = F.adaptive_avg_pool1d(part_features, T_frames)  # (B, 5, T_frames)

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

        # _build_skel_grid có joint_to_part là learnable nên KHÔNG detach ở đây
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
