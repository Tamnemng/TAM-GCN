"""
ResNet + CTR-GCN On-The-Fly V2
================================
Changes from V0:
  1. Conv1d(256->1) + ReLU LEARNABLE replaces fixed L2 norm for skeleton channel projection
     -> Model learns which GCN channels are important (gets gradients!)
  2. Conv1d(20->5,1) LEARNABLE replaces fixed joint indexing for body part grouping
     -> Learns optimal joint weights per part (initialized with anatomical groups)
  3. ConvTranspose2d learned upsampling replaces F.interpolate bilinear (5x5 -> 28x28)
     -> Model learns to map [Body Part x Time] -> [Height x Width] of RGB feature map

Comparison:
  V0: Fixed L2 norm, pick 1 joint/part, bilinear upsample.
  V1: Learnable Conv1d projection, avg all joints/part, bilinear + Conv2d refine.
  V2: Learnable Conv1d projection, learnable Conv1d joint-to-part, ConvTranspose2d upsample.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CrossModalAttentionV2(nn.Module):
    """Cross-Modal Attention V2: with learned ConvTranspose2d upsampling.

    Same CBAM-inspired channel+spatial structure as V0, but replaces
    F.interpolate(bilinear) with learned progressive ConvTranspose2d
    upsampling: 5x5 -> 14x14 -> 28x28.
    """
    def __init__(self, rgb_channels, skel_grid_size=25, reduction=4):
        super().__init__()

        # 1. CHANNEL ATTENTION: skeleton guides channel selection (same as V0)
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False),
            nn.Sigmoid()
        )

        # 2. LEARNED UPSAMPLING: 5x5 -> 14x14 -> 28x28 via ConvTranspose2d
        # Replaces F.interpolate(bilinear) which has no learnable parameters.
        # ConvTranspose2d learns to map [Body Part x Time] -> [H x W] spatial space.
        hidden_ch = 16
        self.skel_upsample = nn.Sequential(
            # Feature extraction on 5x5 grid
            nn.Conv2d(1, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            # 5x5 -> 14x14: output = (5-1)*3 - 2*1 + 4 = 14
            nn.ConvTranspose2d(hidden_ch, hidden_ch, kernel_size=4, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            # 14x14 -> 28x28: output = (14-1)*2 - 2*1 + 4 = 28
            nn.ConvTranspose2d(hidden_ch, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

        # 3. SPATIAL ATTENTION: cross-modal (RGB cues + learned-upsampled skeleton)
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

        # --- STEP 1: CHANNEL ATTENTION ---
        skel_flat = skel_grid.view(B, -1)                       # (B, 25)
        ch_attn = self.channel_attn(skel_flat)                  # (B, C)
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)           # (B, C, 1, 1)
        feat_ca = rgb_feat * ch_attn                            # (B, C, H, W)

        if exp_type == 'no_spatial':
            return rgb_feat + feat_ca

        # --- STEP 2: LEARNED UPSAMPLING (ConvTranspose2d) ---
        # 5x5 -> 14x14 -> 28x28 through learnable transposed convolutions
        skel_sp = self.skel_upsample(skel_grid)                 # (B, 1, 28, 28)

        # RGB spatial cues
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]    # (B, 1, H, W)
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)      # (B, 1, H, W)

        # Cross-modal spatial attention
        sp_input = torch.cat([rgb_max, rgb_avg, skel_sp], dim=1)  # (B, 3, H, W)
        sp_attn = self.spatial_conv(sp_input)                    # (B, 1, H, W)

        # --- STEP 3: MODULATION + RESIDUAL ---
        modulated = feat_ca * sp_attn                            # (B, C, H, W)
        return rgb_feat + modulated


class Model(nn.Module):
    """V2: Learnable skeleton projection + learnable joint-to-part + learned upsampling.

    Key improvements over V0:
      1. skel_proj (Conv1d 256->1 + ReLU): learns which GCN channels matter
         (V0 used fixed L2 norm -> no gradient, all channels weighted equally)
      2. joint_to_part (Conv1d 20->5 + BN + ReLU): learns optimal joint grouping
         (V0 picked only 1 representative joint per part -> 75% joints discarded)
         Initialized with anatomical body part groups for stable training start
      3. skel_upsample (ConvTranspose2d): learns spatial mapping 5x5 -> 28x28
         (V0 used bilinear interpolation -> no spatial semantics)
    """
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5, exp_type='normal'):
        super(Model, self).__init__()

        self.exp_type = exp_type
        self.ctrgcn = ''  # Processor will inject frozen CTR-GCN
        self.temporal_rgb_frames = temporal_rgb_frames

        # ---- ResNet-50 backbone (split into stages) ----
        resnet = models.resnet50(pretrained=pretrained)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )                                                       # -> (B, 64, 56, 56)
        self.layer1 = resnet.layer1                              # -> (B, 256, 56, 56)
        self.layer2 = resnet.layer2                              # -> (B, 512, 28, 28)
        self.layer3 = resnet.layer3                              # -> (B, 1024, 14, 14)
        self.layer4 = resnet.layer4                              # -> (B, 2048, 7, 7)
        self.avgpool = resnet.avgpool                            # -> (B, 2048, 1, 1)
        self.fc = nn.Linear(resnet.fc.in_features, num_class)    # -> (B, num_class)

        # ---- V2: Learnable skeleton channel projection ----
        # Replaces: intensity = (feature_s ** 2).sum(dim=1) ** 0.5  (L2 norm, fixed)
        # With:     Conv1d(256->1) + ReLU that learns which GCN channels are important
        gcn_channels = 256  # CTR-GCN last layer output channels
        self.skel_proj = nn.Sequential(
            nn.Conv1d(gcn_channels, 1, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # ---- V2: Learnable joint-to-part grouping ----
        # Conv1d(20->5, kernel_size=1): each output channel learns to weight 20 input joints
        # Initialized with anatomical body part groups for stable training start
        self.joint_to_part = nn.Sequential(
            nn.Conv1d(20, 5, kernel_size=1, bias=False),
            nn.BatchNorm1d(5),
            nn.ReLU(inplace=True),
        )
        # UCLA 20-joint skeleton layout (0-indexed):
        #   Torso:  0 (hip center), 1 (spine), 2 (neck), 3 (head)
        #   L-Arm:  4 (L shoulder), 5 (L elbow), 6 (L wrist), 7 (L hand)
        #   R-Arm:  8 (R shoulder), 9 (R elbow), 10 (R wrist), 11 (R hand)
        #   L-Leg:  12 (L hip), 13 (L knee), 14 (L ankle), 15 (L foot)
        #   R-Leg:  16 (R hip), 17 (R knee), 18 (R ankle), 19 (R foot)
        part_groups = [
            [0, 1, 2, 3],       # Head/Torso
            [4, 5, 6, 7],       # Left arm
            [8, 9, 10, 11],     # Right arm
            [12, 13, 14, 15],   # Left leg
            [16, 17, 18, 19],   # Right leg
        ]
        with torch.no_grad():
            self.joint_to_part[0].weight.zero_()
            for i, group in enumerate(part_groups):
                for j in group:
                    self.joint_to_part[0].weight[i, j, 0] = 1.0 / len(group)

        # ---- V2: Cross-modal attention with learned upsampling ----
        self.cross_attn = CrossModalAttentionV2(
            rgb_channels=512,
            skel_grid_size=5 * temporal_rgb_frames,  # 5 parts x 5 frames = 25
            reduction=4
        )

    def _build_skel_grid_v2(self, feature_s):
        """V2: Learnable projection -> learnable joint-to-part -> temporal pool.

        Pipeline:
          GCN output (B, 256, T_new, 20, 1)             <- frozen features
            | Conv1d(256, 1, 1) + ReLU                   <- LEARNABLE channel projection
          (B, T_new, 20) per-joint importance
            | Conv1d(20, 5, 1) + BN + ReLU               <- LEARNABLE joint-to-part
          (B, 5, T_new)
            | AdaptiveAvgPool1d -> T_frames
          (B, 5, T_frames)
            | unsqueeze
          (B, 1, 5, 5)
        """
        B, C, T_new, V, M = feature_s.shape
        T_frames = self.temporal_rgb_frames

        # Step 1: Learnable channel projection (256 -> 1)
        feat = feature_s[:, :, :, :, 0]                        # (B, 256, T_new, 20)
        feat = feat.reshape(B, C, T_new * V)                   # (B, 256, T_new*20)
        proj = self.skel_proj(feat)                             # (B, 1, T_new*20)
        proj = proj.reshape(B, T_new, V)                       # (B, T_new, 20)

        # Step 2: Learnable joint-to-part grouping (20 -> 5)
        proj = proj.permute(0, 2, 1)                            # (B, 20, T_new)
        parts = self.joint_to_part(proj)                        # (B, 5, T_new)

        # Step 3: Temporal pooling
        parts = F.adaptive_avg_pool1d(parts, T_frames)          # (B, 5, T_frames)

        return parts.unsqueeze(1)                               # (B, 1, 5, 5)

    def forward(self, x_s, x_rgb):
        # x_s:   (B, 3, 52, 20, 1)  = skeleton sequence
        # x_rgb: (B, 3, 224, 224)   = STROI image

        # ===== 1. Skeleton branch: extract features from frozen GCN =====
        with torch.no_grad():
            _, feature_s = self.ctrgcn.extract_feature(x_s)     # (B, 256, T_new, 20, 1)

        # V2: Learnable pipeline — OUTSIDE torch.no_grad() so Conv1d gets gradients!
        skel_grid = self._build_skel_grid_v2(feature_s.detach())  # (B, 1, 5, 5)

        # ===== 2. RGB branch: ResNet stages + cross-modal injection =====
        x = self.stem(x_rgb)       # (B, 64, 56, 56)
        x = self.layer1(x)         # (B, 256, 56, 56)
        x = self.layer2(x)         # (B, 512, 28, 28)

        # Cross-modal injection (V2: ConvTranspose2d learned upsampling)
        x = self.cross_attn(x, skel_grid, exp_type=self.exp_type)

        x = self.layer3(x)         # (B, 1024, 14, 14)
        x = self.layer4(x)         # (B, 2048, 7, 7)
        x = self.avgpool(x)        # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)    # (B, 2048)
        output = self.fc(x)        # (B, num_class)

        return output
