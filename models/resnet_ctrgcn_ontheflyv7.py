"""V7: V6 + V1's Spatial Refine (separate RGB/skeleton paths + additive fusion).

Changes from V6:
1. Giu nguyen V6's progressive upsampling (5x5 -> Conv -> 14x14 -> Conv -> 28x28)
2. Them V1's spatial_refine: residual Conv refinement tren skeleton map da upsample
   -> skel_sp = ReLU(skel_up + spatial_refine(skel_up))
   -> Hoc duoc pattern "cho nao skeleton manh thi tang/giam attention"
3. Tach rieng RGB spatial path (Conv2d(2,1)) va skeleton path,
   ket hop additive: sp_attn = sigmoid(rgb_logits + skel_sp)
   -> Skeleton truc tiep modulate attention thay vi cat chung roi Conv

Ket qua: Skeleton co kha nang hoc "cho nao skeleton manh -> giam attention RGB"
Do phu thuoc skeleton info -> ensemble voi CTR-GCN giam diversity nhung
tang kha nang cross-modal guidance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CrossModalAttention(nn.Module):
    """Cross-Modal Attention: V6 progressive upsampling + V1 spatial refine."""

    def __init__(self, rgb_channels, skel_grid_size=25, reduction=4):
        super().__init__()

        # 1. CHANNEL ATTENTION (same as V6)
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False),
            nn.Sigmoid()
        )

        # 2. SPATIAL ATTENTION
        # V6: Learned progressive upsampling
        # 5x5 -> 14x14 -> 28x28, moi buoc co Conv2d refine
        self.skel_upsample = nn.Sequential(
            # Stage 1: 5x5, extract features
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.skel_upsample_stage2 = nn.Sequential(
            # Stage 2: 14x14, refine
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.skel_upsample_head = nn.Conv2d(8, 1, kernel_size=1, bias=False)

        # V1's spatial_refine: residual refinement on the upsampled skeleton map
        # Hoc duoc pattern tinh vi hon raw progressive upsampling
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)  # Giu output >= 0 (khong bi flip logic)
        )

        # V1-style: Tach rieng RGB spatial processing (khong cat chung voi skeleton)
        self.rgb_spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1)
            # khong sigmoid o day, de additive combination quyet dinh
        )

    def forward(self, rgb_feat, skel_grid, exp_type='normal'):
        B, C, H, W = rgb_feat.shape

        if exp_type == 'noise':
            skel_grid = torch.randn_like(skel_grid)
        elif exp_type == 'ones':
            skel_grid = torch.ones_like(skel_grid)
        elif exp_type == 'zeros':
            skel_grid = torch.zeros_like(skel_grid)

        # --- Channel Attention (same as V6) ---
        skel_flat = skel_grid.view(B, -1)
        ch_attn = self.channel_attn(skel_flat).unsqueeze(-1).unsqueeze(-1)
        feat_ca = rgb_feat * ch_attn

        if exp_type == 'no_spatial':
            return rgb_feat + feat_ca

        # --- Spatial Attention ---
        # Step 1: V6 progressive upsampling (5x5 -> 14x14 -> 28x28)
        skel_sp = self.skel_upsample(skel_grid)           # (B, 8, 5, 5)
        skel_sp = F.interpolate(skel_sp, size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        skel_sp = self.skel_upsample_stage2(skel_sp)      # (B, 8, 14, 14)
        skel_sp = F.interpolate(skel_sp, size=(H, W), mode='bilinear', align_corners=False)
        skel_sp = self.skel_upsample_head(skel_sp)        # (B, 1, 28, 28)

        # Step 2: V1 spatial_refine (residual refinement)
        skel_sp = F.relu(skel_sp + self.spatial_refine(skel_sp))  # (B, 1, H, W)

        # Step 3: V1-style separate RGB spatial cues
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]   # (B, 1, H, W)
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)     # (B, 1, H, W)
        rgb_sp_feat = torch.cat([rgb_max, rgb_avg], dim=1)     # (B, 2, H, W)
        rgb_sp_logits = self.rgb_spatial_conv(rgb_sp_feat)     # (B, 1, H, W)

        # Step 4: V1-style additive combination
        # Skeleton truc tiep anh huong attention (khong bi Conv(3,1) pha loang)
        sp_attn = torch.sigmoid(rgb_sp_logits + skel_sp)       # (B, 1, H, W)

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

        # Cross-modal attention (V7: V6 progressive upsample + V1 spatial refine)
        self.cross_attn = CrossModalAttention(
            rgb_channels=512,
            skel_grid_size=5 * temporal_rgb_frames,
            reduction=4
        )

        # V6 Conv1d learned joint-to-part
        self.joint_to_part = nn.Sequential(
            nn.Conv1d(20, 5, kernel_size=1, bias=False),
            nn.BatchNorm1d(5),
            nn.ReLU(inplace=True),
        )
        # Init weights theo nhom v6
        part_groups = [
            [0, 1, 2, 3],      # Torso
            [4, 5, 6, 7],      # Left Arm
            [8, 9, 10, 11],    # Right Arm
            [12, 13, 14, 15],  # Left Leg
            [16, 17, 18, 19],  # Right Leg
        ]
        with torch.no_grad():
            self.joint_to_part[0].weight.zero_()
            for i, group in enumerate(part_groups):
                for j in group:
                    self.joint_to_part[0].weight[i, j, 0] = 1.0 / len(group)

    def _build_skel_grid(self, intensity_norm):
        """Build (B, 1, 5, T_frames) spatiotemporal skeleton feature grid."""
        B, T_new, V, M = intensity_norm.shape
        T_frames = self.temporal_rgb_frames

        part_features = intensity_norm[:, :, :, 0]      # (B, T_new, V)
        part_features = part_features.permute(0, 2, 1)  # (B, V=20, T_new)
        part_features = self.joint_to_part(part_features)  # (B, 5, T_new)
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
