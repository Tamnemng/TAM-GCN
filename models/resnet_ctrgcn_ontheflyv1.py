"""
ResNet + CTR-GCN On-The-Fly V1
================================
Changes from V0:
  1. Conv1d(256→1) LEARNABLE replaces fixed L2 norm for skeleton channel projection
     → Model learns which GCN channels are important (gets gradients!)
  2. All 20 joints grouped into 5 body parts by averaging (instead of picking 1 representative)
     → Richer skeleton representation, less information loss
  3. Learned spatial refinement (Conv2d) after bilinear upsampling of skeleton grid
     → Sharper attention maps instead of smooth bilinear artifacts
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CrossModalAttentionV1(nn.Module):
    """Cross-Modal Attention V1: with learned spatial refinement.
    
    Same CBAM-inspired structure as V0, but adds a small Conv network
    to refine the bilinearly upsampled skeleton grid (5×5 → 28×28).
    This produces sharper, more spatially meaningful attention maps.
    """
    def __init__(self, rgb_channels, skel_grid_size=25, reduction=4):
        super().__init__()
        
        # 1. CHANNEL ATTENTION: skeleton guides channel selection
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False),
            nn.Sigmoid()
        )
        
        # 2. SPATIAL REFINEMENT: learned conv to sharpen upsampled skeleton grid
        #    Bilinear 5×5→28×28 is too smooth; this refine network adds local detail
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
        )
        
        # 3. SPATIAL ATTENTION: cross-modal (RGB pooled + refined skeleton)
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
        
        # --- BƯỚC 1: CHANNEL ATTENTION ---
        skel_flat = skel_grid.view(B, -1)                      # (B, 25)
        ch_attn = self.channel_attn(skel_flat)                 # (B, C)
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)          # (B, C, 1, 1)
        feat_ca = rgb_feat * ch_attn                           # (B, C, H, W)
        
        if exp_type == 'no_spatial':
            return rgb_feat + feat_ca
        
        # --- BƯỚC 2: SPATIAL ATTENTION (with learned refinement) ---
        # Bilinear upsample skeleton grid 5×5 → 28×28
        skel_up = F.interpolate(skel_grid, size=(H, W), mode='bilinear', align_corners=False)
        # ★ V1: Refine with learned convs (residual connection)
        skel_sp = skel_up + self.spatial_refine(skel_up)        # (B, 1, H, W)
        
        # RGB spatial cues
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]   # (B, 1, H, W)
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)     # (B, 1, H, W)
        
        # Cross-modal spatial attention
        sp_input = torch.cat([rgb_max, rgb_avg, skel_sp], dim=1) # (B, 3, H, W)
        sp_attn = self.spatial_conv(sp_input)                    # (B, 1, H, W)
        
        # --- BƯỚC 3: MODULATION + RESIDUAL ---
        modulated = feat_ca * sp_attn                            # (B, C, H, W)
        return rgb_feat + modulated


class Model(nn.Module):
    """V1: Learnable skeleton projection + body part grouping.
    
    Key improvements over V0:
      1. skel_proj (Conv1d): learns which of 256 GCN channels matter → trainable!
         (V0 used fixed L2 norm → no gradient, all channels weighted equally)
      2. part_groups: averages ALL joints in each body region
         (V0 picked only 1 representative joint per part → 75% joints discarded)
      3. spatial_refine: learned upsampling refinement
         (V0 used raw bilinear → overly smooth attention maps)
    """
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5, exp_type='normal'):
        super(Model, self).__init__()
        
        self.exp_type = exp_type
        self.ctrgcn = ''  # Processor will inject frozen CTR-GCN here
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
        
        # ──── ★ V1: Learnable skeleton channel projection ────
        # Replaces: intensity = (feature_s ** 2).sum(dim=1) ** 0.5  (L2 norm, fixed)
        # With:     Conv1d(256→1) that learns which GCN channels are important
        gcn_channels = 256  # CTR-GCN last layer output channels
        self.skel_proj = nn.Sequential(
            nn.Conv1d(gcn_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm1d(1),
            nn.Sigmoid()                                         # Output ∈ [0, 1]
        )
        
        # ──── Cross-modal attention V1 (with spatial refinement) ────
        self.cross_attn = CrossModalAttentionV1(
            rgb_channels=512,
            skel_grid_size=5 * temporal_rgb_frames,  # 5 parts × 5 frames = 25
            reduction=4
        )
        
        # ──── ★ V1: Body part GROUPS (average all joints, not just 1 representative) ────
        # UCLA 20-joint skeleton layout (0-indexed):
        #   Torso:  0 (hip center), 1 (spine), 2 (neck), 3 (head)
        #   L-Arm:  4 (L shoulder), 5 (L elbow), 6 (L wrist), 7 (L hand)
        #   R-Arm:  8 (R shoulder), 9 (R elbow), 10 (R wrist), 11 (R hand)
        #   L-Leg:  12 (L hip), 13 (L knee), 14 (L ankle), 15 (L foot)
        #   R-Leg:  16 (R hip), 17 (R knee), 18 (R ankle), 19 (R foot)
        self.part_groups = [
            [0, 1, 2, 3],       # Head/Torso — 4 joints averaged
            [4, 5, 6, 7],       # Left arm   — 4 joints averaged
            [8, 9, 10, 11],     # Right arm  — 4 joints averaged
            [12, 13, 14, 15],   # Left leg   — 4 joints averaged
            [16, 17, 18, 19]    # Right leg  — 4 joints averaged
        ]

    def _build_skel_grid_v1(self, feature_s):
        """V1: Learnable projection + body part grouping.
        
        Pipeline:
          GCN output (B, 256, 13, 20, 1)                         ← frozen features
            ↓ Conv1d(256, 1, 1) + BN + Sigmoid                   ← LEARNABLE (has gradients!)
          (B, 1, 13, 20) = per-joint, per-timestep importance
            ↓ Group 20 joints → 5 body parts (avg)               ← uses ALL joints
          (B, 5, 13)
            ↓ Adaptive pool 13 → T_frames temporal bins
          (B, 5, T_frames)
            ↓ unsqueeze
          (B, 1, 5, 5)                                            ← same shape, matches STROI
        """
        B, C, T_new, V, M = feature_s.shape
        T_frames = self.temporal_rgb_frames
        
        # Squeeze M=1 and flatten spatial-temporal dims for Conv1d
        feat = feature_s[:, :, :, :, 0]                       # (B, 256, 13, 20)
        feat = feat.reshape(B, C, T_new * V)                  # (B, 256, 260)
        
        # ★ Learnable projection: 256 channels → 1 importance score
        #   Unlike L2 norm, this LEARNS which channels matter and HAS GRADIENTS
        proj = self.skel_proj(feat)                            # (B, 1, 260)
        proj = proj.reshape(B, T_new, V)                      # (B, 13, 20)
        
        # ★ Group 20 joints → 5 body parts (average within each group)
        #   Unlike V0 which picks 1 joint, this uses ALL joints in each region
        parts = []
        for group in self.part_groups:
            part_feat = proj[:, :, group].mean(dim=2)          # (B, 13)
            parts.append(part_feat)
        parts = torch.stack(parts, dim=1)                      # (B, 5, 13)
        
        # Pool 13 GCN temporal steps → T_frames bins (match STROI columns)
        parts = F.adaptive_avg_pool1d(parts, T_frames)         # (B, 5, 5)
        
        return parts.unsqueeze(1)                              # (B, 1, 5, 5)

    def forward(self, x_s, x_rgb):
        # x_s:   (B, 3, 52, 20, 1)  = skeleton sequence
        # x_rgb: (B, 3, 224, 224)   = STROI image (5×5 body-part grid)
        
        # ===== 1. Skeleton branch: extract features from frozen GCN =====
        with torch.no_grad():
            _, feature_s = self.ctrgcn.extract_feature(x_s)    # (B, 256, 13, 20, 1)
        
        # ★ V1: Learnable projection — OUTSIDE torch.no_grad() so Conv1d gets gradients!
        skel_grid = self._build_skel_grid_v1(feature_s.detach())  # (B, 1, 5, 5)
        
        # ===== 2. RGB branch: ResNet stages + cross-modal injection =====
        x = self.stem(x_rgb)       # (B, 64, 56, 56)
        x = self.layer1(x)         # (B, 256, 56, 56)
        x = self.layer2(x)         # (B, 512, 28, 28)
        
        # ★ Cross-modal injection (V1: with learned spatial refinement)
        x = self.cross_attn(x, skel_grid, exp_type=self.exp_type)
        
        x = self.layer3(x)         # (B, 1024, 14, 14)
        x = self.layer4(x)         # (B, 2048, 7, 7)
        x = self.avgpool(x)        # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)    # (B, 2048)
        output = self.fc(x)        # (B, num_class)
        
        return output
