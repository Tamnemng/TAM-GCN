"""
ResNet + CTR-GCN On-The-Fly V7
================================
Changes from V2:
  1. MULTI-CHANNEL PROJECTION: Conv1d(256->K) instead of Conv1d(256->1)
     -> Preserves intrinsic spatial information of skeleton (K=8 channels retained)
     -> Each of K channels captures a different GCN spatial pattern
     -> Then Conv1d(K->1) aggregates with learned weights (2-stage projection)
     V2 problem: Conv1d(256->1) collapses ALL spatial info into 1 scalar per joint.
                 Different graph conv patterns (local vs global, center vs limb) are lost.
     V7 fix:    256 -> K (preserve diversity) -> 1 (informed aggregation)

  2. CONFIDENCE GATING: Learnable gate that suppresses noisy skeleton input
     -> gate = sigmoid(MLP(skeleton_descriptor))  in [0, 1]
     -> output = rgb + gate * modulated  (instead of rgb + modulated)
     -> When skeleton is noisy/unreliable, gate -> 0, model falls back to pure RGB
     -> When skeleton is informative, gate -> 1, full cross-modal fusion
     V2 problem: No mechanism to reject bad skeleton input. Noisy skeleton
                 actively hurts performance (76.29% vs 87.50%).
     V7 fix:    Hadamard gating allows graceful degradation.

  3. Inherits from V2: Learnable joint-to-part (Conv1d 20->5) + ConvTranspose2d upsample.

Comparison:
  V0: Fixed L2 norm, pick 1 joint/part, bilinear upsample. No gating.
  V2: Conv1d(256->1), Conv1d(20->5), ConvTranspose2d upsample. No gating.
  V7: Conv1d(256->K->1) 2-stage, Conv1d(20->5), ConvTranspose2d upsample. Confidence gating.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConfidenceGate(nn.Module):
    """Learnable confidence gate for skeleton reliability estimation.

    Takes the skeleton descriptor (flattened skel_grid) and outputs a scalar
    gate value in [0, 1] via sigmoid. When skeleton is noisy/uninformative,
    gate approaches 0 and the model falls back to pure RGB (residual path).

    Architecture: Linear(25 -> 16) -> ReLU -> Linear(16 -> 1) -> Sigmoid
    """
    def __init__(self, skel_grid_size=25):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(skel_grid_size, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1, bias=True),
        )
        # Initialize bias to +1.0 so initial gate ~ sigmoid(1) ~ 0.73
        # This ensures the model starts by USING skeleton (not ignoring it)
        # and only learns to suppress when necessary.
        nn.init.constant_(self.gate_mlp[2].bias, 1.0)

    def forward(self, skel_flat):
        """
        Args:
            skel_flat: (B, skel_grid_size) flattened skeleton grid
        Returns:
            gate: (B, 1, 1, 1) broadcast-ready gate value in [0, 1]
        """
        g = self.gate_mlp(skel_flat)           # (B, 1)
        g = torch.sigmoid(g)                    # (B, 1) in [0, 1]
        return g.unsqueeze(-1).unsqueeze(-1)    # (B, 1, 1, 1)


class CrossModalAttentionV7(nn.Module):
    """Cross-Modal Attention V7: with confidence gating + learned upsampling.

    Same CBAM-inspired channel+spatial structure as V2, but adds:
    - Confidence gate: output = rgb + gate * modulated
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

        # 2. LEARNED UPSAMPLING: 5x5 -> 14x14 -> 28x28 via ConvTranspose2d
        hidden_ch = 16
        self.skel_upsample = nn.Sequential(
            nn.Conv2d(1, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            # 5x5 -> 14x14
            nn.ConvTranspose2d(hidden_ch, hidden_ch, kernel_size=4, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(hidden_ch, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

        # 3. SPATIAL ATTENTION: cross-modal (RGB cues + learned-upsampled skeleton)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # 4. CONFIDENCE GATE: learns when to trust skeleton
        self.confidence_gate = ConfidenceGate(skel_grid_size)

    def forward(self, rgb_feat, skel_grid, exp_type='normal'):
        B, C, H, W = rgb_feat.shape

        # ABLATION experiments
        if exp_type == 'noise':
            skel_grid = torch.randn_like(skel_grid)
        elif exp_type == 'ones':
            skel_grid = torch.ones_like(skel_grid)
        elif exp_type == 'zeros':
            skel_grid = torch.zeros_like(skel_grid)

        # --- CONFIDENCE GATE ---
        skel_flat = skel_grid.view(B, -1)                       # (B, 25)
        gate = self.confidence_gate(skel_flat)                   # (B, 1, 1, 1)

        # --- STEP 1: CHANNEL ATTENTION ---
        ch_attn = self.channel_attn(skel_flat)                   # (B, C)
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)            # (B, C, 1, 1)
        feat_ca = rgb_feat * ch_attn                             # (B, C, H, W)

        if exp_type == 'no_spatial':
            return rgb_feat + gate * feat_ca

        # --- STEP 2: LEARNED UPSAMPLING (ConvTranspose2d) ---
        skel_sp = self.skel_upsample(skel_grid)                 # (B, 1, 28, 28)

        # RGB spatial cues
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]    # (B, 1, H, W)
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)      # (B, 1, H, W)

        # Cross-modal spatial attention
        sp_input = torch.cat([rgb_max, rgb_avg, skel_sp], dim=1)  # (B, 3, H, W)
        sp_attn = self.spatial_conv(sp_input)                    # (B, 1, H, W)

        # --- STEP 3: GATED MODULATION + RESIDUAL ---
        modulated = feat_ca * sp_attn                            # (B, C, H, W)
        return rgb_feat + gate * modulated                       # gate controls skeleton influence


class Model(nn.Module):
    """V7: 2-stage skeleton projection + confidence gating + learned upsampling.

    Key improvements over V2:
      1. skel_proj: Conv1d(256->K) + Conv1d(K->1) two-stage projection
         - Stage 1 (256->K): preserves K diverse spatial patterns from GCN
         - Stage 2 (K->1): informed aggregation of K retained patterns
         V2 used single Conv1d(256->1) which collapses all spatial info at once.

      2. confidence_gate: MLP(25->16->1) + Sigmoid
         - Learns to suppress skeleton when it's noisy/unreliable
         - gate * modulated: Hadamard product allows graceful degradation
         V2 had no mechanism to reject bad skeleton input.

      3. Inherits from V2: Conv1d(20->5) joint-to-part + ConvTranspose2d upsample.
    """
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5,
                 exp_type='normal', proj_channels=8):
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

        # ---- V7: TWO-STAGE skeleton channel projection ----
        # Stage 1: Conv1d(256 -> K) — retain K diverse GCN spatial patterns
        # Stage 2: Conv1d(K -> 1) — informed aggregation
        #
        # WHY 2 stages? Conv1d(256->1) in V2 forces the model to make a single
        # linear combination of 256 channels. With 2 stages:
        #   - Stage 1 acts as a "bottleneck" that preserves K most important patterns
        #   - BatchNorm + ReLU between stages adds nonlinearity
        #   - Stage 2 can see the preserved patterns and make a better decision
        gcn_channels = 256
        K = proj_channels  # default 8
        self.skel_proj = nn.Sequential(
            nn.Conv1d(gcn_channels, K, kernel_size=1, bias=False),
            nn.BatchNorm1d(K),
            nn.ReLU(inplace=True),
            nn.Conv1d(K, 1, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # ---- V7: Learnable joint-to-part grouping (same as V2) ----
        self.joint_to_part = nn.Sequential(
            nn.Conv1d(20, 5, kernel_size=1, bias=False),
            nn.BatchNorm1d(5),
            nn.ReLU(inplace=True),
        )
        # UCLA 20-joint skeleton layout (0-indexed):
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

        # ---- V7: Cross-modal attention with confidence gating ----
        self.cross_attn = CrossModalAttentionV7(
            rgb_channels=512,
            skel_grid_size=5 * temporal_rgb_frames,  # 5 parts x 5 frames = 25
            reduction=4
        )

    def _build_skel_grid_v7(self, feature_s):
        """V7: 2-stage projection -> learnable joint-to-part -> temporal pool.

        Pipeline:
          GCN output (B, 256, T_new, 20, 1)             <- frozen features
            | Conv1d(256, K, 1) + BN + ReLU              <- STAGE 1: retain K patterns
            | Conv1d(K, 1, 1) + ReLU                     <- STAGE 2: aggregate
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

        # Step 1: Two-stage channel projection (256 -> K -> 1)
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

        # V7: 2-stage projection — OUTSIDE torch.no_grad() so Conv1d gets gradients!
        skel_grid = self._build_skel_grid_v7(feature_s.detach())  # (B, 1, 5, 5)

        # ===== 2. RGB branch: ResNet stages + gated cross-modal injection =====
        x = self.stem(x_rgb)       # (B, 64, 56, 56)
        x = self.layer1(x)         # (B, 256, 56, 56)
        x = self.layer2(x)         # (B, 512, 28, 28)

        # Cross-modal injection (V7: confidence-gated)
        x = self.cross_attn(x, skel_grid, exp_type=self.exp_type)

        x = self.layer3(x)         # (B, 1024, 14, 14)
        x = self.layer4(x)         # (B, 2048, 7, 7)
        x = self.avgpool(x)        # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)    # (B, 2048)
        output = self.fc(x)        # (B, num_class)

        return output
