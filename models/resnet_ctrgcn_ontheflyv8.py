"""
ResNet + CTR-GCN On-The-Fly V8
================================
Focus: Fix the NEAR-UNIFORM attention map problem of V2/V7.

Root cause analysis:
  V2/V7 attention entropy = 9.563 vs uniform = 9.615 (gap only 0.052 bit)
  Inter-class attention similarity = 0.973 (almost identical across classes!)

  WHY? Three compounding factors:
    1. ConvTranspose2d 5x5→28x28 produces inherently smooth output
    2. Single Conv2d 7x7 + Sigmoid has limited capacity to sharpen
    3. Sigmoid output ∈ [0,1] with natural bias toward 0.5 → near-uniform

V8 fixes with THREE targeted changes:

  1. SPATIAL-WISE GATING (replaces V7's scalar gate)
     V7: gate = scalar (B, 1, 1, 1) → same confidence everywhere
     V8: gate = spatial map (B, 1, H, W) → per-pixel skeleton confidence
     -> Skeleton can be reliable for arms region but unreliable for legs
     -> Different spatial regions get different gate values

  2. TEMPERATURE-SHARPENED ATTENTION
     V2: Sigmoid(x) → output clusters around 0.5 when x is small
     V8: Sigmoid(x / τ) with LEARNABLE temperature τ, initialized to 0.3
     -> Small τ → steeper sigmoid → sharper attention map
     -> τ is learnable so model finds optimal sharpness during training

  3. DEEPER SPATIAL NETWORK (replaces single 7x7 conv)
     V2: Conv2d(3→1, 7x7) + BN + Sigmoid → 1 layer, limited capacity
     V8: Conv2d(3→8, 3x3) + BN + ReLU → Conv2d(8→8, 3x3) + BN + ReLU
         → Conv2d(8→1, 1x1) + TempSigmoid
     -> 3 layers with 3x3 kernels can learn SHARPER spatial patterns
     -> Same receptive field as 7x7 (3+3+1=7) but more nonlinearity

Comparison:
  V0: Fixed L2, pick 1 joint, bilinear, no gate. Entropy: not measured.
  V2: Conv1d(256→1), ConvTranspose2d, 7x7+Sigmoid. Entropy gap: 0.052 bit.
  V7: 2-stage proj, ConvTranspose2d, 7x7+Sigmoid, scalar gate.
  V8: 2-stage proj, ConvTranspose2d, deep 3x3+TempSigmoid, spatial-wise gate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialWiseGate(nn.Module):
    """Spatial-wise gating: per-pixel skeleton confidence.

    Instead of a single scalar gate (V7), produces a spatial gate map
    (B, 1, H, W) so different regions can have different skeleton confidence.

    Architecture:
      skeleton grid (B, 1, 5, 5)
        → ConvTranspose2d upsample to (B, 1, 28, 28)
        → Conv2d refinement
        → Sigmoid → gate_map (B, 1, H, W)

    The gate map learns WHERE skeleton information is trustworthy:
      - High gate near body regions (skeleton is accurate there)
      - Low gate in background (skeleton has no info there)
    """
    def __init__(self):
        super().__init__()
        hidden_ch = 8
        self.gate_net = nn.Sequential(
            # Feature extraction on 5x5
            nn.Conv2d(1, hidden_ch, kernel_size=3, padding=1, bias=False),
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
        # The final conv bias in ConvTranspose2d is False, so we rely on
        # BatchNorm bias to set initial operating point
        nn.init.constant_(self.gate_net[-2].bias, 0.85)  # BN bias → sigmoid(0.85) ≈ 0.7

    def forward(self, skel_grid):
        """
        Args:
            skel_grid: (B, 1, 5, 5) skeleton feature grid
        Returns:
            gate_map: (B, 1, H, W) per-pixel gate in [0, 1]
        """
        return self.gate_net(skel_grid)  # (B, 1, 28, 28)


class CrossModalAttentionV8(nn.Module):
    """Cross-Modal Attention V8: sharp spatial attention + spatial-wise gating.

    Key differences from V2/V7:
      1. Deeper spatial network (3-layer 3x3 instead of 1-layer 7x7)
      2. Temperature-sharpened sigmoid for attention
      3. Spatial-wise gating instead of scalar gate
    """
    def __init__(self, rgb_channels, skel_grid_size=25, reduction=4,
                 init_temperature=0.3):
        super().__init__()

        # 1. CHANNEL ATTENTION: skeleton guides channel selection (same as V2)
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False),
            nn.Sigmoid()
        )

        # 2. LEARNED UPSAMPLING: 5x5 → 14x14 → 28x28 (same as V2)
        hidden_ch = 16
        self.skel_upsample = nn.Sequential(
            nn.Conv2d(1, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_ch, hidden_ch, kernel_size=4, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_ch, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

        # 3. DEEP SPATIAL ATTENTION with temperature-sharpened sigmoid
        # V2: Conv2d(3, 1, 7x7) + BN + Sigmoid → 1 layer, limited capacity, near-uniform
        # V8: 3-layer network with 3x3 kernels → same RF but sharper patterns
        sp_hidden = 8
        self.spatial_net = nn.Sequential(
            nn.Conv2d(3, sp_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sp_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(sp_hidden, sp_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sp_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(sp_hidden, 1, kernel_size=1, bias=True),
            # No sigmoid here — applied with temperature below
        )

        # LEARNABLE temperature: controls attention sharpness
        # τ < 1 → sharper sigmoid, τ > 1 → smoother
        # Initialized to 0.3 for sharp attention, learned during training
        # Stored as log to ensure τ > 0 via exp()
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())

        # 4. SPATIAL-WISE GATE: per-pixel skeleton confidence
        self.spatial_gate = SpatialWiseGate()

    def forward(self, rgb_feat, skel_grid, exp_type='normal'):
        B, C, H, W = rgb_feat.shape

        # ABLATION experiments
        if exp_type == 'noise':
            skel_grid = torch.randn_like(skel_grid)
        elif exp_type == 'ones':
            skel_grid = torch.ones_like(skel_grid)
        elif exp_type == 'zeros':
            skel_grid = torch.zeros_like(skel_grid)

        skel_flat = skel_grid.view(B, -1)                       # (B, 25)

        # --- SPATIAL-WISE GATE ---
        gate_map = self.spatial_gate(skel_grid)                  # (B, 1, H, W)

        # --- STEP 1: CHANNEL ATTENTION ---
        ch_attn = self.channel_attn(skel_flat)                   # (B, C)
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)            # (B, C, 1, 1)
        feat_ca = rgb_feat * ch_attn                             # (B, C, H, W)

        if exp_type == 'no_spatial':
            # Use mean of gate_map as scalar fallback
            gate_scalar = gate_map.mean(dim=[2, 3], keepdim=True)  # (B, 1, 1, 1)
            return rgb_feat + gate_scalar * feat_ca

        # --- STEP 2: LEARNED UPSAMPLING ---
        skel_sp = self.skel_upsample(skel_grid)                 # (B, 1, 28, 28)

        # RGB spatial cues
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]    # (B, 1, H, W)
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)      # (B, 1, H, W)

        # --- STEP 3: DEEP SPATIAL ATTENTION with temperature ---
        sp_input = torch.cat([rgb_max, rgb_avg, skel_sp], dim=1)  # (B, 3, H, W)
        sp_logits = self.spatial_net(sp_input)                   # (B, 1, H, W)

        # Temperature-sharpened sigmoid: σ(x / τ)
        temperature = self.log_temperature.exp()                 # scalar > 0
        sp_attn = torch.sigmoid(sp_logits / temperature)         # (B, 1, H, W)

        # --- STEP 4: SPATIAL-GATED MODULATION + RESIDUAL ---
        modulated = feat_ca * sp_attn                            # (B, C, H, W)
        return rgb_feat + gate_map * modulated                   # spatial-wise gating


class Model(nn.Module):
    """V8: Sharp spatial attention + spatial-wise gating.

    Builds on V7 (2-stage projection + gating) and fixes the attention problem:
      1. skel_proj: Conv1d(256→K→1) two-stage (from V7)
      2. joint_to_part: Conv1d(20→5) learnable (from V2)
      3. NEW: Deep spatial network (3-layer 3x3) replaces single 7x7
      4. NEW: Temperature-sharpened sigmoid for attention sharpness
      5. NEW: Spatial-wise gate (B, 1, H, W) replaces scalar gate (B, 1, 1, 1)
    """
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5,
                 exp_type='normal', proj_channels=8, init_temperature=0.3):
        super(Model, self).__init__()

        self.exp_type = exp_type
        self.ctrgcn = ''
        self.temporal_rgb_frames = temporal_rgb_frames

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

        # ---- Two-stage skeleton projection (from V7) ----
        gcn_channels = 256
        K = proj_channels
        self.skel_proj = nn.Sequential(
            nn.Conv1d(gcn_channels, K, kernel_size=1, bias=False),
            nn.BatchNorm1d(K),
            nn.ReLU(inplace=True),
            nn.Conv1d(K, 1, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # ---- Learnable joint-to-part grouping (from V2) ----
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

        # ---- V8: Cross-modal attention with sharp spatial + spatial-wise gate ----
        self.cross_attn = CrossModalAttentionV8(
            rgb_channels=512,
            skel_grid_size=5 * temporal_rgb_frames,
            reduction=4,
            init_temperature=init_temperature,
        )

    def _build_skel_grid(self, feature_s):
        """2-stage projection → joint-to-part → temporal pool."""
        B, C, T_new, V, M = feature_s.shape
        T_frames = self.temporal_rgb_frames

        feat = feature_s[:, :, :, :, 0]                        # (B, 256, T_new, 20)
        feat = feat.reshape(B, C, T_new * V)                   # (B, 256, T_new*20)
        proj = self.skel_proj(feat)                             # (B, 1, T_new*20)
        proj = proj.reshape(B, T_new, V)                       # (B, T_new, 20)

        proj = proj.permute(0, 2, 1)                            # (B, 20, T_new)
        parts = self.joint_to_part(proj)                        # (B, 5, T_new)
        parts = F.adaptive_avg_pool1d(parts, T_frames)          # (B, 5, T_frames)

        return parts.unsqueeze(1)                               # (B, 1, 5, 5)

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
