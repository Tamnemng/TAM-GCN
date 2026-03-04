"""
ResNet + CTR-GCN On-The-Fly V2: Decoupled Cross-Modal Attention
================================================================
Goal: Skeleton guides WHERE to look, RGB decides WHAT features to use.
       → Strong standalone + diverse from CTR-GCN → best ensemble.

Key Design:
  - Channel Attention: Pure RGB self-attention (CBAM-style).
    RGB selects its own important channels WITHOUT skeleton influence.
    → Develops complementary representations, diverse from CTR-GCN.
    
  - Spatial Attention: Cross-modal (skeleton + RGB pooled features).
    Skeleton tells the model WHERE to pay attention in the feature map.
    → Gets skeleton guidance for spatial focus.
    
  - Skeleton Grid: Fixed L2 norm (no learnable projection, avoids mimicking GCN)
    + All-joints body part grouping (richer than V0's single representative).

Comparison:
  V0: Skeleton → Channel + Spatial (both). Fixed L2, 1 joint/part.
      → Diverse ensemble (96.98%), weak standalone (80.17%)
  V1: Skeleton → Channel + Spatial (both). Learnable Conv1d, all joints.
      → Strong standalone (89.66%), weak ensemble (95.91%)
  V2: Skeleton → Spatial ONLY. RGB → Channel ONLY. Fixed L2, all joints.
      → Aim: Strong standalone + diverse ensemble (best of both worlds)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoupledCrossModalAttention(nn.Module):
    """Decoupled Cross-Modal Attention: Skeleton guides spatial, RGB owns channels.
    
    Unlike V0/V1 where skeleton controls both channel and spatial attention,
    V2 separates concerns:
      - Channel attention is RGB-only (CBAM self-attention)
      - Spatial attention is cross-modal (skeleton provides WHERE signal)
    This ensures RGB develops its own feature representation (good for ensemble)
    while still benefiting from skeleton spatial guidance (good for standalone).
    """
    def __init__(self, rgb_channels, reduction=16):
        super().__init__()
        
        # ──── 1. CHANNEL ATTENTION: Pure RGB self-attention (CBAM-style) ────
        # RGB decides which channels are important based on ITS OWN features.
        # No skeleton involvement → RGB develops independent channel representation.
        self.ch_mlp = nn.Sequential(
            nn.Linear(rgb_channels, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False)
        )
        # Shared MLP applied to both max-pooled and avg-pooled features
        
        # ──── 2. SPATIAL ATTENTION: Cross-modal (skeleton guides WHERE) ────
        # Skeleton provides spatial map: which regions in the image are important.
        # Combined with RGB's own spatial cues (max/avg pool).
        
        # Learned refinement for upsampled skeleton grid (5×5→28×28)
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
        )
        
        # Final spatial conv: RGB cues + skeleton spatial map → attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, skel_grid, exp_type='normal'):
        B, C, H, W = rgb_feat.shape
        
        # ABLATION experiments
        if exp_type == 'noise':
            skel_grid = torch.randn_like(skel_grid)
        elif exp_type == 'ones':
            skel_grid = torch.ones_like(skel_grid)
        elif exp_type == 'zeros':
            skel_grid = torch.zeros_like(skel_grid)
        
        # ──── BƯỚC 1: RGB SELF-CHANNEL ATTENTION (CBAM-style) ────
        # Max and Avg pool across spatial dims → (B, C, 1, 1)
        max_pool = F.adaptive_max_pool2d(rgb_feat, 1).squeeze(-1).squeeze(-1)   # (B, C)
        avg_pool = F.adaptive_avg_pool2d(rgb_feat, 1).squeeze(-1).squeeze(-1)   # (B, C)
        
        # Shared MLP on both, then combine → channel attention weights
        ch_attn = torch.sigmoid(self.ch_mlp(max_pool) + self.ch_mlp(avg_pool))  # (B, C)
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)                           # (B, C, 1, 1)
        
        # Apply channel attention
        feat_ca = rgb_feat * ch_attn                                            # (B, C, H, W)
        
        # EXPERIMENT: Channel-only baseline
        if exp_type == 'no_spatial':
            return rgb_feat + feat_ca
        
        # ──── BƯỚC 2: CROSS-MODAL SPATIAL ATTENTION (skeleton guides WHERE) ────
        # Bilinear upsample skeleton grid: 5×5 → H×W (28×28)
        skel_up = F.interpolate(skel_grid, size=(H, W), mode='bilinear', align_corners=False)
        # Learned refinement (residual) → sharper spatial map
        skel_sp = skel_up + self.spatial_refine(skel_up)                        # (B, 1, H, W)
        
        # RGB spatial cues (from channel-attended features)
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]                   # (B, 1, H, W)
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)                     # (B, 1, H, W)
        
        # Cross-modal spatial attention: RGB cues + skeleton WHERE signal
        sp_input = torch.cat([rgb_max, rgb_avg, skel_sp], dim=1)               # (B, 3, H, W)
        sp_attn = self.spatial_conv(sp_input)                                   # (B, 1, H, W)
        
        # ──── BƯỚC 3: MODULATION + RESIDUAL ────
        modulated = feat_ca * sp_attn                                           # (B, C, H, W)
        return rgb_feat + modulated


class Model(nn.Module):
    """V2: Decoupled Cross-Modal Attention.
    
    Skeleton → spatial guidance only (WHERE to look).
    RGB → channel selection independently (WHAT features to use).
    
    This ensures:
      ✓ Skeleton still guides attention → improved standalone over pure ResNet
      ✓ RGB develops OWN channel representation → diverse from CTR-GCN
      ✓ Diverse + strong = best ensemble performance
    """
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5, exp_type='normal'):
        super(Model, self).__init__()
        
        self.exp_type = exp_type
        self.ctrgcn = ''  # Processor will inject frozen CTR-GCN
        self.temporal_rgb_frames = temporal_rgb_frames
        
        # ──── ResNet-50 backbone (split into stages) ────
        resnet = models.resnet50(pretrained=pretrained)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )                                                       # → (B, 64, 56, 56)
        self.layer1 = resnet.layer1                              # → (B, 256, 56, 56)
        self.layer2 = resnet.layer2                              # → (B, 512, 28, 28)
        self.layer3 = resnet.layer3                              # → (B, 1024, 14, 14)
        self.layer4 = resnet.layer4                              # → (B, 2048, 7, 7)
        self.avgpool = resnet.avgpool                            # → (B, 2048, 1, 1)
        self.fc = nn.Linear(resnet.fc.in_features, num_class)    # → (B, num_class)
        
        # ──── Decoupled Cross-Modal Attention ────
        self.cross_attn = DecoupledCrossModalAttention(
            rgb_channels=512,
            reduction=16
        )
        
        # ──── Body part groups: ALL joints averaged (from V1) ────
        # UCLA 20-joint skeleton:
        #   0-3: Head/Torso, 4-7: L-Arm, 8-11: R-Arm, 12-15: L-Leg, 16-19: R-Leg
        self.part_groups = [
            [0, 1, 2, 3],       # Head/Torso
            [4, 5, 6, 7],       # Left arm
            [8, 9, 10, 11],     # Right arm
            [12, 13, 14, 15],   # Left leg
            [16, 17, 18, 19]    # Right leg
        ]

    def _build_skel_grid(self, intensity_norm):
        """Build skeleton grid using FIXED L2 norm + all-joints grouping.
        
        Uses V0's fixed approach (no learnable projection → avoids mimicking GCN)
        but with V1's all-joints grouping (richer than single representative).
        
        Pipeline:
          intensity_norm (B, T_new, V, M) — already L2 normed + min-max normalized
            ↓ Group 20 joints → 5 body parts (average within each group)
          (B, 5, T_new)
            ↓ Adaptive pool T_new → T_frames
          (B, 5, T_frames)
            ↓ Normalize + unsqueeze
          (B, 1, 5, T_frames) = (B, 1, 5, 5)  — matches STROI layout
        """
        B, T_new, V, M = intensity_norm.shape
        T_frames = self.temporal_rgb_frames
        
        # Group ALL joints → 5 body parts (average)
        part_features = []
        for group in self.part_groups:
            # Average intensity across all joints in this body part
            group_feat = intensity_norm[:, :, group, 0].mean(dim=2)  # (B, T_new)
            part_features.append(group_feat)
        part_features = torch.stack(part_features, dim=1)            # (B, 5, T_new)
        
        # Adaptive pool T_new → T_frames temporal bins
        part_features = F.adaptive_avg_pool1d(part_features, T_frames)  # (B, 5, T_frames)
        
        # Normalize to [0, 1] range
        part_features = part_features / 127.0
        
        return part_features.unsqueeze(1)  # (B, 1, 5, T_frames) = (B, 1, 5, 5)

    def forward(self, x_s, x_rgb):
        # x_s:   (B, 3, 52, 20, 1) = skeleton sequence
        # x_rgb: (B, 3, 224, 224)  = STROI image
        
        # ===== 1. Skeleton branch: extract features (frozen, fixed) =====
        with torch.no_grad():
            _, feature_s = self.ctrgcn.extract_feature(x_s)
            
            # Fixed L2 norm across channel dim (NOT learnable → preserves diversity)
            intensity = (feature_s * feature_s).sum(dim=1) ** 0.5
            intensity = torch.abs(intensity)
            
            # Per-sample min-max normalize to [0, 255]
            B = intensity.shape[0]
            flat = intensity.view(B, -1)
            f_min = flat.min(dim=1, keepdim=True)[0]
            f_max = flat.max(dim=1, keepdim=True)[0]
            diff = f_max - f_min
            diff[diff == 0] = 1e-6
            flat_norm = 255.0 * (flat - f_min) / diff
            intensity_norm = flat_norm.view_as(intensity)
        
        # Build skeleton spatial grid (fixed processing, no learnable params)
        skel_grid = self._build_skel_grid(intensity_norm.detach())  # (B, 1, 5, 5)
        
        # ===== 2. RGB branch: ResNet stages + decoupled cross-modal attention =====
        x = self.stem(x_rgb)       # (B, 64, 56, 56)
        x = self.layer1(x)         # (B, 256, 56, 56)
        x = self.layer2(x)         # (B, 512, 28, 28)
        
        # ★ Decoupled injection: skeleton → spatial only, RGB → channel only
        x = self.cross_attn(x, skel_grid, exp_type=self.exp_type)
        
        x = self.layer3(x)         # (B, 1024, 14, 14)
        x = self.layer4(x)         # (B, 2048, 7, 7)
        x = self.avgpool(x)        # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)    # (B, 2048)
        output = self.fc(x)        # (B, num_class)
        
        return output
