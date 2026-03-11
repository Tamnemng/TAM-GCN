"""
ResNet + CTR-GCN On-The-Fly V10
================================
Focus: Maximize ENSEMBLE diversity while keeping V9's multi-channel benefits.

Problem analysis:
  V9 achieves best single-model accuracy (88.58%) and attention quality,
  BUT ensemble with CTR-GCN is WORSE than V8 ensemble (95.91% vs 96.55%).

  WHY? V9's K-channel skeleton grid makes it too CORRELATED with CTR-GCN:
    - V9 preserves K=8 channels of skeleton dynamics => relies heavily on skeleton
    - When skeleton info is wrong, BOTH V9 and CTR-GCN fail on the same samples
    - Ensemble needs DIVERSITY: models should fail on DIFFERENT samples
    - V8's compressed 1-channel grid forced it to rely more on RGB => more diverse

  The DIVERSITY-ACCURACY trade-off:
    V8: weaker skeleton use => more RGB reliance => different errors from CTR-GCN
    V9: stronger skeleton use => less RGB reliance => same errors as CTR-GCN

V10 fix: SKELETON PATH DROPOUT — force RGB independence during training.

  Key insight: During training, randomly zero out the ENTIRE skeleton grid
  with probability p. This forces the model to:
    (a) When skeleton is present (1-p): use rich K-channel features (V9 benefit)
    (b) When skeleton is dropped (p): classify using RGB alone (diversity benefit)

  At test time: skeleton is ALWAYS used (no dropout).

  This is analogous to classic Dropout preventing neuron co-adaptation:
    - Here we prevent RGB-skeleton CO-ADAPTATION
    - RGB backbone learns to be strong independently
    - Skeleton path learns to ENHANCE, not REPLACE, RGB features

  Expected outcome:
    - Single model: comparable or slightly lower than V9 (RGB must work harder)
    - Ensemble: HIGHER than both V8 and V9 (more diverse from CTR-GCN)
    - Noise robustness: BETTER (RGB fallback when skeleton is noisy)

  Additional: CHANNEL-WISE DROPOUT on skeleton grid
    - Even when skeleton is not fully dropped, randomly mask individual channels
    - Prevents model from relying on any single skeleton dynamics pattern
    - Encourages redundant representation across channels

Comparison:
  V0:  Fixed L2, pick 1 joint, bilinear, no gate.
  V2:  Conv1d(256->1), skel_grid (B,1,5,5), 7x7+Sigmoid.
  V8:  Conv1d(256->K->1), skel_grid (B,1,5,5), deep+TempSigmoid, spatial gate.
  V9:  Conv1d(256->K), skel_grid (B,K,5,5), deep+TempSigmoid, spatial gate.
  V10: Conv1d(256->K), skel_grid (B,K,5,5), deep+TempSigmoid, spatial gate,
       + skeleton path dropout (p=0.3) + channel dropout (p=0.2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialWiseGate(nn.Module):
    """Spatial-wise gating from K-channel skeleton grid (same as V9)."""
    def __init__(self, skel_channels=8):
        super().__init__()
        hidden_ch = 8
        self.gate_net = nn.Sequential(
            nn.Conv2d(skel_channels, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_ch, hidden_ch, kernel_size=4, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_ch, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_net[-2].bias, 0.85)

    def forward(self, skel_grid):
        return self.gate_net(skel_grid)


class CrossModalAttentionV10(nn.Module):
    """Cross-Modal Attention V10: V9 architecture + skeleton path dropout.

    Same architecture as V9, but with training-time dropout on skeleton path:
      1. path_dropout_p: probability of zeroing out ENTIRE skeleton grid
      2. channel_dropout_p: probability of zeroing out individual K channels

    This forces RGB backbone to be strong independently => ensemble diversity.
    """
    def __init__(self, rgb_channels, skel_channels=8, skel_grid_size=200,
                 reduction=4, init_temperature=0.3, sp_skel_channels=4,
                 path_dropout_p=0.3, channel_dropout_p=0.2):
        super().__init__()

        self.sp_skel_channels = sp_skel_channels
        self.path_dropout_p = path_dropout_p
        self.channel_dropout_p = channel_dropout_p
        self.skel_channels = skel_channels

        # 1. CHANNEL ATTENTION (same as V9)
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False),
            nn.Sigmoid()
        )

        # 2. MULTI-CHANNEL UPSAMPLING (same as V9)
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

        # 3. DEEP SPATIAL ATTENTION (same as V9)
        sp_hidden = 8
        self.spatial_net = nn.Sequential(
            nn.Conv2d(2 + sp_skel_channels, sp_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sp_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(sp_hidden, sp_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sp_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(sp_hidden, 1, kernel_size=1, bias=True),
        )

        # LEARNABLE temperature (same as V9)
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())

        # 4. SPATIAL-WISE GATE (same as V9)
        self.spatial_gate = SpatialWiseGate(skel_channels)

    def _apply_skeleton_dropout(self, skel_grid):
        """Apply skeleton path dropout during training.

        Two levels of dropout:
          1. PATH dropout: zero out entire grid with prob path_dropout_p
             -> Forces RGB-only classification for these samples
          2. CHANNEL dropout: zero out random K channels with prob channel_dropout_p
             -> Prevents reliance on any single dynamics pattern

        At test time: no dropout (full skeleton info always used).
        """
        if not self.training:
            return skel_grid

        B = skel_grid.shape[0]
        device = skel_grid.device

        # Level 1: Path dropout — zero out entire skeleton for some samples
        # Bernoulli mask: 1 = keep, 0 = drop
        path_mask = torch.bernoulli(
            torch.full((B, 1, 1, 1), 1.0 - self.path_dropout_p, device=device)
        )
        # Scale by 1/(1-p) to maintain expected value (inverted dropout)
        if self.path_dropout_p < 1.0:
            skel_grid = skel_grid * path_mask / (1.0 - self.path_dropout_p)

        # Level 2: Channel dropout — zero out random channels for surviving samples
        # Only apply if path wasn't fully dropped
        if self.channel_dropout_p > 0:
            K = self.skel_channels
            ch_mask = torch.bernoulli(
                torch.full((B, K, 1, 1), 1.0 - self.channel_dropout_p, device=device)
            )
            if self.channel_dropout_p < 1.0:
                skel_grid = skel_grid * ch_mask / (1.0 - self.channel_dropout_p)

        return skel_grid

    def forward(self, rgb_feat, skel_grid, exp_type='normal'):
        B, C, H, W = rgb_feat.shape

        # ABLATION experiments
        if exp_type == 'noise':
            skel_grid = torch.randn_like(skel_grid)
        elif exp_type == 'ones':
            skel_grid = torch.ones_like(skel_grid)
        elif exp_type == 'zeros':
            skel_grid = torch.zeros_like(skel_grid)

        # V10: Apply skeleton dropout during training
        skel_grid = self._apply_skeleton_dropout(skel_grid)

        skel_flat = skel_grid.view(B, -1)                       # (B, K*25)

        # --- SPATIAL-WISE GATE ---
        gate_map = self.spatial_gate(skel_grid)                  # (B, 1, H, W)

        # --- STEP 1: CHANNEL ATTENTION ---
        ch_attn = self.channel_attn(skel_flat)                   # (B, C)
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)            # (B, C, 1, 1)
        feat_ca = rgb_feat * ch_attn                             # (B, C, H, W)

        if exp_type == 'no_spatial':
            gate_scalar = gate_map.mean(dim=[2, 3], keepdim=True)
            return rgb_feat + gate_scalar * feat_ca

        # --- STEP 2: MULTI-CHANNEL UPSAMPLING ---
        skel_sp = self.skel_upsample(skel_grid)                 # (B, K_sp, 28, 28)

        # RGB spatial cues
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]    # (B, 1, H, W)
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)      # (B, 1, H, W)

        # --- STEP 3: DEEP SPATIAL ATTENTION with temperature ---
        sp_input = torch.cat([rgb_max, rgb_avg, skel_sp], dim=1)  # (B, 2+K_sp, H, W)
        sp_logits = self.spatial_net(sp_input)                   # (B, 1, H, W)

        temperature = self.log_temperature.exp()
        sp_attn = torch.sigmoid(sp_logits / temperature)         # (B, 1, H, W)

        # --- STEP 4: SPATIAL-GATED MODULATION + RESIDUAL ---
        modulated = feat_ca * sp_attn                            # (B, C, H, W)
        return rgb_feat + gate_map * modulated


class Model(nn.Module):
    """V10: V9 + Skeleton Path Dropout for ensemble diversity.

    Architecture identical to V9 (multi-channel skeleton projection),
    plus training-time skeleton dropout to force RGB independence:
      1. skel_proj: Conv1d(256->K) preserves K channels (from V9)
      2. joint_to_part: Conv1d(20->5) per K-channel (from V9)
      3. skel_grid: (B, K, 5, 5) multi-channel (from V9)
      4. Deep spatial + TempSigmoid + spatial gate (from V9/V8)
      5. NEW: path_dropout_p=0.3 — 30% chance of zeroing skeleton entirely
      6. NEW: channel_dropout_p=0.2 — 20% chance of zeroing each K channel
    """
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5,
                 exp_type='normal', proj_channels=8, init_temperature=0.3,
                 sp_skel_channels=4, path_dropout_p=0.3, channel_dropout_p=0.2):
        super(Model, self).__init__()

        self.exp_type = exp_type
        self.ctrgcn = ''
        self.temporal_rgb_frames = temporal_rgb_frames
        self.proj_channels = proj_channels

        # ---- ResNet-50 backbone ----
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

        # ---- Single-stage projection preserving K channels (from V9) ----
        gcn_channels = 256
        K = proj_channels
        self.skel_proj = nn.Sequential(
            nn.Conv1d(gcn_channels, K, kernel_size=1, bias=False),
            nn.BatchNorm1d(K),
            nn.ReLU(inplace=True),
        )

        # ---- Learnable joint-to-part grouping (from V9) ----
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

        # ---- V10: Cross-modal attention with skeleton dropout ----
        self.cross_attn = CrossModalAttentionV10(
            rgb_channels=512,
            skel_channels=K,
            skel_grid_size=K * 5 * temporal_rgb_frames,
            reduction=4,
            init_temperature=init_temperature,
            sp_skel_channels=sp_skel_channels,
            path_dropout_p=path_dropout_p,
            channel_dropout_p=channel_dropout_p,
        )

    def _build_skel_grid(self, feature_s):
        """Single-stage projection -> per-channel joint-to-part -> temporal pool.
        Identical to V9.
        """
        B, C, T_new, V, M = feature_s.shape
        K = self.proj_channels
        T_frames = self.temporal_rgb_frames

        feat = feature_s[:, :, :, :, 0]                        # (B, 256, T_new, 20)
        feat = feat.reshape(B, C, T_new * V)                   # (B, 256, T_new*20)

        proj = self.skel_proj(feat)                             # (B, K, T_new*20)
        proj = proj.reshape(B, K, T_new, V)                    # (B, K, T_new, 20)

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
