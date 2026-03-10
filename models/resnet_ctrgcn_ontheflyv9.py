"""
ResNet + CTR-GCN On-The-Fly V9
================================
Focus: Break the INFORMATION BOTTLENECK in skeleton projection.

Root cause analysis (V2→V8 shared problem):
  V2/V7/V8 all use skel_proj that collapses 256 → 1 channel:
    V2: Conv1d(256→1)        — direct collapse, loses everything
    V7: Conv1d(256→K→1)      — two-stage but still collapses to 1
    V8: Conv1d(256→K→1)      — same as V7, just better attention

  WHY is this a problem?
    1. CTR-GCN's 256 channels encode RICH dynamics: velocity, joint angles,
       trajectory curvature, relative bone orientations, etc.
    2. Collapsing to 1 channel forces the model to discard almost everything,
       keeping only a scalar "importance score" per joint-frame.
    3. joint_to_part grouping on scalar values can only group by intensity,
       NOT by shared motion patterns (e.g., wrist+elbow same velocity pattern).
    4. skel_upsample from (B, 1, 5, 5) has extremely limited spatial info
       to guide attention — just 1 feature map for the entire body.

V9 fix: PRESERVE K channels throughout the entire skeleton pipeline.

  1. MULTI-CHANNEL PROJECTION: Conv1d(256→K) preserves K feature patterns
     V8: Conv1d(256→K→1) → skel_grid (B, 1, 5, 5) — scalar per body part
     V9: Conv1d(256→K)   → skel_grid (B, K, 5, 5) — vector per body part
     -> Each channel captures a DIFFERENT dynamics aspect (velocity, angle, etc.)
     -> Information capacity: K*25 = 200 values vs 25 values (8x more)

  2. RICHER JOINT-TO-PART GROUPING: Conv1d(20→5) applied per K-channel
     V8: Groups 20 joints → 5 parts using 1 scalar per joint
     V9: Groups using K-dim feature vectors per joint
     -> Can detect joints with SIMILAR motion patterns across channels
     -> e.g., wrist+elbow share velocity pattern in channel 3 → group together

  3. MULTI-CHANNEL SPATIAL GUIDANCE
     V8: skel_upsample (B, 1, 5, 5) → (B, 1, 28, 28) — 1 spatial map
     V9: skel_upsample (B, K, 5, 5) → (B, K_sp, 28, 28) — multiple maps
     -> spatial_net receives 2 + K_sp channels instead of 2 + 1
     -> Each skel spatial map highlights DIFFERENT body part dynamics
     -> Richer input → more discriminative attention

  4. K-CHANNEL SPATIAL-WISE GATE
     V8: SpatialWiseGate input (B, 1, 5, 5) — only knows "where is skeleton"
     V9: SpatialWiseGate input (B, K, 5, 5) — knows "what is skeleton doing"
     -> Gate can be high for RELEVANT dynamics and low for IRRELEVANT ones

  Keeps V8's three innovations:
    - Deep spatial network (3-layer 3x3)
    - Temperature-sharpened sigmoid
    - Spatial-wise gating

Comparison:
  V0: Fixed L2, pick 1 joint, bilinear, no gate.
  V2: Conv1d(256→1), skel_grid (B,1,5,5), 7x7+Sigmoid.
  V7: Conv1d(256→K→1), skel_grid (B,1,5,5), 7x7+Sigmoid, scalar gate.
  V8: Conv1d(256→K→1), skel_grid (B,1,5,5), deep 3x3+TempSigmoid, spatial gate.
  V9: Conv1d(256→K),   skel_grid (B,K,5,5), deep 3x3+TempSigmoid, spatial gate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialWiseGate(nn.Module):
    """Spatial-wise gating from K-channel skeleton grid.

    V8: Input (B, 1, 5, 5) — only knows spatial position of skeleton.
    V9: Input (B, K, 5, 5) — knows both position AND dynamics of skeleton.
         Richer input → gate can distinguish relevant vs irrelevant dynamics.

    Architecture:
      skeleton grid (B, K, 5, 5)
        → Conv2d feature extraction on 5x5
        → ConvTranspose2d upsample to 14x14 → 28x28
        → Sigmoid → gate_map (B, 1, H, W)
    """
    def __init__(self, skel_channels=8):
        super().__init__()
        hidden_ch = 8
        self.gate_net = nn.Sequential(
            # Feature extraction on 5x5 (input: K channels)
            nn.Conv2d(skel_channels, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            # 5x5 → 14x14
            nn.ConvTranspose2d(hidden_ch, hidden_ch, kernel_size=4, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            # 14x14 → 28x28
            nn.ConvTranspose2d(hidden_ch, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        # Initialize so gate starts around 0.7 (use skeleton by default)
        nn.init.constant_(self.gate_net[-2].bias, 0.85)  # BN bias → sigmoid(0.85) ≈ 0.7

    def forward(self, skel_grid):
        """
        Args:
            skel_grid: (B, K, 5, 5) multi-channel skeleton feature grid
        Returns:
            gate_map: (B, 1, H, W) per-pixel gate in [0, 1]
        """
        return self.gate_net(skel_grid)  # (B, 1, 28, 28)


class CrossModalAttentionV9(nn.Module):
    """Cross-Modal Attention V9: multi-channel skeleton + sharp attention + spatial gate.

    Key differences from V8:
      1. Channel attention input: K*25 features (vs 25) — richer skeleton encoding
      2. Spatial upsample: K channels → K_sp spatial maps (vs 1 → 1)
      3. Spatial net input: 2 + K_sp channels (vs 2 + 1) — more skeleton guidance
      4. Gate input: K channels (vs 1) — dynamics-aware gating

    Keeps from V8:
      - Deep 3-layer spatial network
      - Temperature-sharpened sigmoid
      - Spatial-wise (per-pixel) gating
    """
    def __init__(self, rgb_channels, skel_channels=8, skel_grid_size=200,
                 reduction=4, init_temperature=0.3, sp_skel_channels=4):
        super().__init__()

        self.sp_skel_channels = sp_skel_channels

        # 1. CHANNEL ATTENTION: K*25 skeleton features guide channel selection
        # V8: Linear(25→128→512)  — only "where is skeleton"
        # V9: Linear(K*25→128→512) — "where AND what is skeleton doing"
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False),
            nn.Sigmoid()
        )

        # 2. MULTI-CHANNEL UPSAMPLING: (B, K, 5, 5) → (B, K_sp, 28, 28)
        # V8: Conv2d(1→16→16→1) — 1 spatial map
        # V9: Conv2d(K→16→16→K_sp) — K_sp spatial maps, each highlighting
        #     different dynamics (e.g., velocity map, angle map, trajectory map)
        hidden_ch = 16
        self.skel_upsample = nn.Sequential(
            nn.Conv2d(skel_channels, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_ch, hidden_ch, kernel_size=4, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_ch, sp_skel_channels, kernel_size=4, stride=2, padding=1, bias=False),
        )

        # 3. DEEP SPATIAL ATTENTION with richer input
        # V8: Cat(rgb_max, rgb_avg, skel_sp) → (B, 3, H, W)
        # V9: Cat(rgb_max, rgb_avg, skel_sp_K) → (B, 2+K_sp, H, W)
        sp_hidden = 8
        self.spatial_net = nn.Sequential(
            nn.Conv2d(2 + sp_skel_channels, sp_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sp_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(sp_hidden, sp_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sp_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(sp_hidden, 1, kernel_size=1, bias=True),
            # No sigmoid here — applied with temperature below
        )

        # LEARNABLE temperature (from V8)
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())

        # 4. SPATIAL-WISE GATE from K-channel grid (from V8, but K input channels)
        self.spatial_gate = SpatialWiseGate(skel_channels)

    def forward(self, rgb_feat, skel_grid, exp_type='normal'):
        B, C, H, W = rgb_feat.shape

        # ABLATION experiments
        if exp_type == 'noise':
            skel_grid = torch.randn_like(skel_grid)
        elif exp_type == 'ones':
            skel_grid = torch.ones_like(skel_grid)
        elif exp_type == 'zeros':
            skel_grid = torch.zeros_like(skel_grid)

        skel_flat = skel_grid.view(B, -1)                       # (B, K*25)

        # --- SPATIAL-WISE GATE (from K-channel grid) ---
        gate_map = self.spatial_gate(skel_grid)                  # (B, 1, H, W)

        # --- STEP 1: CHANNEL ATTENTION ---
        ch_attn = self.channel_attn(skel_flat)                   # (B, C)
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)            # (B, C, 1, 1)
        feat_ca = rgb_feat * ch_attn                             # (B, C, H, W)

        if exp_type == 'no_spatial':
            gate_scalar = gate_map.mean(dim=[2, 3], keepdim=True)  # (B, 1, 1, 1)
            return rgb_feat + gate_scalar * feat_ca

        # --- STEP 2: MULTI-CHANNEL UPSAMPLING ---
        skel_sp = self.skel_upsample(skel_grid)                 # (B, K_sp, 28, 28)

        # RGB spatial cues
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]    # (B, 1, H, W)
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)      # (B, 1, H, W)

        # --- STEP 3: DEEP SPATIAL ATTENTION with temperature ---
        sp_input = torch.cat([rgb_max, rgb_avg, skel_sp], dim=1)  # (B, 2+K_sp, H, W)
        sp_logits = self.spatial_net(sp_input)                   # (B, 1, H, W)

        # Temperature-sharpened sigmoid: σ(x / τ)
        temperature = self.log_temperature.exp()                 # scalar > 0
        sp_attn = torch.sigmoid(sp_logits / temperature)         # (B, 1, H, W)

        # --- STEP 4: SPATIAL-GATED MODULATION + RESIDUAL ---
        modulated = feat_ca * sp_attn                            # (B, C, H, W)
        return rgb_feat + gate_map * modulated                   # spatial-wise gating


class Model(nn.Module):
    """V9: Multi-channel skeleton projection + sharp attention + spatial gate.

    Key change from V8: skel_proj outputs K channels (not collapsed to 1).
      1. skel_proj: Conv1d(256→K) — preserves K feature channels (NEW)
      2. joint_to_part: Conv1d(20→5) applied per K-channel via reshape (ADAPTED)
      3. skel_grid: (B, K, 5, 5) multi-channel (was (B, 1, 5, 5) in V8)
      4. Deep spatial network with K_sp spatial skeleton maps (from V8, enhanced)
      5. Temperature-sharpened sigmoid (from V8)
      6. Spatial-wise gate from K-channel grid (from V8, enhanced)
    """
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5,
                 exp_type='normal', proj_channels=8, init_temperature=0.3,
                 sp_skel_channels=4):
        super(Model, self).__init__()

        self.exp_type = exp_type
        self.ctrgcn = ''
        self.temporal_rgb_frames = temporal_rgb_frames
        self.proj_channels = proj_channels

        # ---- ResNet-50 backbone ----
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

        # ---- V9: Single-stage projection preserving K channels ----
        # V8: Conv1d(256→K→1) two-stage, collapses to scalar
        # V9: Conv1d(256→K)   single-stage, preserves K feature patterns
        gcn_channels = 256
        K = proj_channels
        self.skel_proj = nn.Sequential(
            nn.Conv1d(gcn_channels, K, kernel_size=1, bias=False),
            nn.BatchNorm1d(K),
            nn.ReLU(inplace=True),
        )

        # ---- Learnable joint-to-part grouping (from V2, adapted for K channels) ----
        # Applied per K-channel: (B*K, 20, T_new) → (B*K, 5, T_new)
        self.joint_to_part = nn.Sequential(
            nn.Conv1d(20, 5, kernel_size=1, bias=False),
            nn.BatchNorm1d(5),
            nn.ReLU(inplace=True),
        )
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

        # ---- V9: Cross-modal attention with multi-channel skeleton ----
        self.cross_attn = CrossModalAttentionV9(
            rgb_channels=512,
            skel_channels=K,
            skel_grid_size=K * 5 * temporal_rgb_frames,   # K*25 = 200
            reduction=4,
            init_temperature=init_temperature,
            sp_skel_channels=sp_skel_channels,
        )

    def _build_skel_grid(self, feature_s):
        """Single-stage projection → per-channel joint-to-part → temporal pool.

        V8 pipeline: (B,256,T,20) → proj (B,1,T*20) → reshape (B,T,20)
                     → permute (B,20,T) → parts (B,5,T) → pool (B,5,5) → (B,1,5,5)

        V9 pipeline: (B,256,T,20) → proj (B,K,T*20) → reshape (B,K,T,20)
                     → permute (B,K,20,T) → reshape (B*K,20,T) → parts (B*K,5,T)
                     → pool (B*K,5,5) → reshape (B,K,5,5)
        """
        B, C, T_new, V, M = feature_s.shape
        K = self.proj_channels
        T_frames = self.temporal_rgb_frames

        feat = feature_s[:, :, :, :, 0]                        # (B, 256, T_new, 20)
        feat = feat.reshape(B, C, T_new * V)                   # (B, 256, T_new*20)

        # V9: project 256 → K, KEEP all K channels
        proj = self.skel_proj(feat)                             # (B, K, T_new*20)
        proj = proj.reshape(B, K, T_new, V)                    # (B, K, T_new, 20)

        # joint_to_part per K-channel via reshape trick
        proj = proj.permute(0, 1, 3, 2)                        # (B, K, 20, T_new)
        proj = proj.reshape(B * K, V, T_new)                   # (B*K, 20, T_new)
        parts = self.joint_to_part(proj)                        # (B*K, 5, T_new)
        parts = F.adaptive_avg_pool1d(parts, T_frames)          # (B*K, 5, T_frames)
        parts = parts.reshape(B, K, 5, T_frames)               # (B, K, 5, 5)

        return parts                                            # (B, K, 5, 5)

    def forward(self, x_s, x_rgb):
        with torch.no_grad():
            _, feature_s = self.ctrgcn.extract_feature(x_s)

        skel_grid = self._build_skel_grid(feature_s.detach())

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
