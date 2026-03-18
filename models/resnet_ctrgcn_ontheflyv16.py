"""
ResNet + CTR-GCN On-The-Fly V16
================================
V15 + learnable temperature on spatial attention (re-added from V11).

Analysis of V15 vs V11 ensemble gap (96.77% vs 96.98%):
  Root cause: V11 has `sp_attn = sigmoid(logits / T)` with learnable T.
  This temperature sharpens/softens spatial attention adaptively, producing
  output distributions more diverse from CTR-GCN → higher ensemble gain.
  V15 removed temperature → sp_attn distributions closer to CTR-GCN → less diversity.

V16 = V15 + log_temperature (learnable, init=0.3 same as V11).
  All V15 improvements retained:
    [V13] Y-flip, normalized jitter, confidence gate input fix
    [V14] SpatialWiseGate, multi-frame coordinate extraction
    [V15] skel_upsample replaces flat cross_out, sigma_coarse 8→4,
          confidence bias 1.73→0.5

Single change from V15:
  sp_attn = sigmoid(spatial_net(sp_input) / temperature)   ← V16
  sp_attn = sigmoid(spatial_net(sp_input))                 ← V15

Expected effect:
  - Model learns to sharpen sp_attn for discriminative actions (T→0)
    or soften for ambiguous ones (T→∞)
  - More diverse error distribution vs CTR-GCN → better ensemble
  - Standalone: neutral to slightly positive (T regularizes spatial attention)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialWiseGate(nn.Module):
    """V11 transposed-conv spatial gate (kept from V14)."""
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
        return self.gate_net(skel_grid)  # (B, 1, H, W)


class CrossModalAttentionV16(nn.Module):
    """Cross-Modal Attention V16.

    Spatial attention paths (3 paths feeding spatial_net):
      Path A: scatter_sharp  — Gaussian heatmaps (sigma~3px), localized
      Path B: scatter_coarse — Gaussian heatmaps (sigma~4px), moderately spread
      Path C: skel_upsample  — ConvTranspose of skel_grid, spatially varying
               (replaces V13/V14's sigmoid cross-attention which was spatially flat)

    Additional gates:
      SpatialWiseGate: second multiplicative spatial gate (V14 fix)
      Confidence gate: [skel_flat, rgb_global] → scalar α (V13 fix, lower init bias)

    sp_input_channels = 2 (rgb_max/avg) + 4 (scatter_sharp) +
                        4 (scatter_coarse) + 4 (skel_up) = 14  (same as V14)
    """
    def __init__(self, rgb_channels, skel_channels=8, skel_grid_size=200,
                 reduction=4, num_parts=5, sp_feat_channels=4,
                 init_sigma_sharp=3.0, init_sigma_coarse=4.0,
                 cross_attn_dim=16, init_temperature=0.3):
        super().__init__()

        self.num_parts = num_parts
        self.sp_feat_channels = sp_feat_channels

        # ── CHANNEL ATTENTION ──────────────────────────────────────────────
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False),
            nn.Sigmoid()
        )

        # ── PATH A/B: HEATMAP-GUIDED FEATURE SCATTER ──────────────────────
        self.part_feat_proj = nn.Sequential(
            nn.Conv1d(skel_channels, sp_feat_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(sp_feat_channels),
            nn.ReLU(inplace=True),
        )
        self.coord_adjust = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
        )
        nn.init.zeros_(self.coord_adjust[-1].weight)
        nn.init.zeros_(self.coord_adjust[-1].bias)

        self.log_sigma_sharp  = nn.Parameter(torch.tensor(init_sigma_sharp).log())
        # BUG B FIX: coarse sigma reduced from 8.0 → 4.0
        self.log_sigma_coarse = nn.Parameter(torch.tensor(init_sigma_coarse).log())

        # ── LEARNABLE TEMPERATURE (V16: re-added from V11) ────────────────
        # sp_attn = sigmoid(logits / T); T learnable, init=0.3
        # Low T  → sharp attention (focus on top pixels)
        # High T → soft attention (spread across more pixels)
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())

        # ── PATH C: SKELETON UPSAMPLE (replaces cross-attn) ───────────────
        # BUG A FIX: replace sigmoid cross-attention (spatially uniform) with
        # ConvTranspose upsampling (spatially varying). Direct learned mapping
        # from skeleton temporal dynamics to a 28×28 spatial feature map.
        hidden_up = 16
        self.skel_upsample = nn.Sequential(
            nn.Conv2d(skel_channels, hidden_up, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_up),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_up, hidden_up, kernel_size=4, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_up),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_up, sp_feat_channels, kernel_size=4, stride=2, padding=1, bias=False),
        )

        # ── SPATIAL ATTENTION NETWORK ──────────────────────────────────────
        # Input: rgb_max(1) + rgb_avg(1) + scatter_sharp(C_sp) +
        #        scatter_coarse(C_sp) + skel_up(C_sp) = 2 + 3*C_sp = 14
        sp_input_channels = 2 + 3 * sp_feat_channels
        sp_hidden = 8
        self.spatial_net = nn.Sequential(
            nn.Conv2d(sp_input_channels, sp_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sp_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(sp_hidden, sp_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sp_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(sp_hidden, 1, kernel_size=1, bias=True),
        )

        # ── SPATIAL-WISE GATE (V14 fix, kept) ─────────────────────────────
        self.spatial_gate = SpatialWiseGate(skel_channels)

        # ── CONFIDENCE GATE ────────────────────────────────────────────────
        # BUG C FIX: lower initial bias (1.73 → 0.5) to reduce saturation.
        # sigmoid(0.5) ≈ 0.62 starting confidence — leaves room for gate to
        # adapt per-sample rather than staying pinned near 0.97.
        confidence_input_dim = skel_grid_size + rgb_channels
        confidence_hidden = 64
        self.confidence_gate = nn.Sequential(
            nn.Linear(confidence_input_dim, confidence_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(confidence_hidden, 1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.confidence_gate[-2].bias, 0.5)   # was 1.73 in V13/V14
        nn.init.xavier_uniform_(self.confidence_gate[-2].weight, gain=0.1)
        nn.init.xavier_uniform_(self.confidence_gate[0].weight, gain=0.5)

    @staticmethod
    def _generate_gaussian_heatmaps(part_coords, H, W, sigma):
        B, P, _ = part_coords.shape
        device = part_coords.device
        cx = (part_coords[:, :, 0] + 1) / 2 * (W - 1)
        cy = (part_coords[:, :, 1] + 1) / 2 * (H - 1)
        yy = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
        cx = cx.view(B, P, 1, 1)
        cy = cy.view(B, P, 1, 1)
        return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    def forward(self, rgb_feat, skel_grid, part_coords, exp_type='normal'):
        B, C, H, W = rgb_feat.shape

        if exp_type == 'noise':
            skel_grid = torch.randn_like(skel_grid)
            part_coords = torch.rand_like(part_coords) * 2 - 1
        elif exp_type == 'ones':
            skel_grid = torch.ones_like(skel_grid)
        elif exp_type == 'zeros':
            skel_grid = torch.zeros_like(skel_grid)

        skel_flat = skel_grid.view(B, -1)

        # ── CONFIDENCE GATE ────────────────────────────────────────────────
        rgb_global = F.adaptive_avg_pool2d(rgb_feat, 1).view(B, -1)
        confidence = self.confidence_gate(torch.cat([skel_flat, rgb_global], dim=1))

        # ── CHANNEL ATTENTION ──────────────────────────────────────────────
        ch_attn = self.channel_attn(skel_flat).unsqueeze(-1).unsqueeze(-1)
        feat_ca = rgb_feat * ch_attn

        if exp_type == 'no_spatial':
            alpha = confidence.unsqueeze(-1).unsqueeze(-1)
            return rgb_feat + alpha * feat_ca, confidence

        # ── SPATIAL-WISE GATE (V14) ────────────────────────────────────────
        gate_map = self.spatial_gate(skel_grid)   # (B, 1, H, W)

        # ── PATH A/B: GAUSSIAN SCATTER ────────────────────────────────────
        skel_parts     = skel_grid.mean(dim=3)                       # (B, K, 5)
        skel_parts_proj = self.part_feat_proj(skel_parts)            # (B, C_sp, 5)

        adj_coords = (part_coords + self.coord_adjust(part_coords)).clamp(-1, 1)

        sigma_sharp  = self.log_sigma_sharp.exp()
        sigma_coarse = self.log_sigma_coarse.exp()
        hm_sharp  = self._generate_gaussian_heatmaps(adj_coords, H, W, sigma_sharp)
        hm_coarse = self._generate_gaussian_heatmaps(adj_coords, H, W, sigma_coarse)

        scatter_sharp  = torch.einsum('bcp,bphw->bchw', skel_parts_proj, hm_sharp)
        scatter_coarse = torch.einsum('bcp,bphw->bchw', skel_parts_proj, hm_coarse)

        # ── PATH C: SKELETON UPSAMPLE (BUG A FIX) ────────────────────────
        # ConvTranspose maps skel_grid (B, K, 5, 5) → (B, C_sp, ~28, ~28)
        # then center-crop / adaptive pool to match exact (H, W)
        skel_up = self.skel_upsample(skel_grid)                      # (B, C_sp, H', W')
        if skel_up.shape[2:] != (H, W):
            skel_up = F.adaptive_avg_pool2d(skel_up, (H, W))

        # ── SPATIAL ATTENTION ─────────────────────────────────────────────
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)

        sp_input = torch.cat([
            rgb_max, rgb_avg,
            scatter_sharp,
            scatter_coarse,
            skel_up,                                                  # replaces cross_out
        ], dim=1)

        temperature = self.log_temperature.exp()
        sp_attn = torch.sigmoid(self.spatial_net(sp_input) / temperature)  # (B, 1, H, W)

        # ── FUSION ────────────────────────────────────────────────────────
        skel_delta = feat_ca * sp_attn * gate_map
        alpha = confidence.unsqueeze(-1).unsqueeze(-1)
        output_feat = rgb_feat + alpha * skel_delta

        return output_feat, confidence


class Model(nn.Module):
    """V16: V15 + learnable temperature on spatial attention (from V11).

    Single change vs V15:
      sp_attn = sigmoid(spatial_net(sp_input) / temperature)
      where log_temperature is learnable, init=0.3 (same as V11).
    """
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5,
                 exp_type='normal', proj_channels=8, sp_feat_channels=4,
                 init_sigma_sharp=3.0, init_sigma_coarse=4.0,
                 init_temperature=0.3, cross_attn_dim=16, consistency_weight=0.1,
                 num_point=20, num_person=1):
        super(Model, self).__init__()

        self.exp_type = exp_type
        self.ctrgcn = ''
        self.temporal_rgb_frames = temporal_rgb_frames
        self.proj_channels = proj_channels
        self.consistency_weight = consistency_weight
        self.num_class = num_class
        self.num_person = num_person
        self.num_point = num_point

        if num_point == 25:
            self.part_groups = [
                [0, 1, 2, 3, 20], [4, 5, 6, 7, 21, 22],
                [8, 9, 10, 11, 23, 24], [12, 13, 14, 15], [16, 17, 18, 19],
            ]
        else:
            self.part_groups = [
                [0, 1, 2, 3], [4, 5, 6, 7],
                [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19],
            ]

        resnet = models.resnet50(pretrained=pretrained)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(resnet.fc.in_features, num_class)

        self.rgb_only_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(512, num_class),
        )

        gcn_channels = 256
        K = proj_channels
        self.skel_proj = nn.Sequential(
            nn.Conv1d(gcn_channels, K, kernel_size=1, bias=False),
            nn.BatchNorm1d(K), nn.ReLU(inplace=True),
        )
        self.joint_to_part = nn.Sequential(
            nn.Conv1d(num_point, 5, kernel_size=1, bias=False),
            nn.BatchNorm1d(5), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            self.joint_to_part[0].weight.zero_()
            for i, group in enumerate(self.part_groups):
                for j in group:
                    self.joint_to_part[0].weight[i, j, 0] = 1.0 / len(group)

        self.cross_attn = CrossModalAttentionV16(
            rgb_channels=512, skel_channels=K,
            skel_grid_size=K * 5 * temporal_rgb_frames,
            reduction=4, num_parts=5,
            sp_feat_channels=sp_feat_channels,
            init_sigma_sharp=init_sigma_sharp,
            init_sigma_coarse=init_sigma_coarse,
            init_temperature=init_temperature,
            cross_attn_dim=cross_attn_dim,
        )

    def _build_skel_grid(self, feature_s):
        B, C, T_new, V, M = feature_s.shape
        K, T_frames = self.proj_channels, self.temporal_rgb_frames
        feat = feature_s.mean(dim=4) if M > 1 else feature_s[:, :, :, :, 0]
        feat = feat.reshape(B, C, T_new * V)
        proj = self.skel_proj(feat).reshape(B, K, T_new, V)
        proj = proj.permute(0, 1, 3, 2).reshape(B * K, V, T_new)
        parts = self.joint_to_part(proj)
        parts = F.adaptive_avg_pool1d(parts, T_frames)
        return parts.reshape(B, K, 5, T_frames)

    def _extract_part_coords(self, x_s):
        """Multi-frame averaged coordinates (V14 fix), with Y-flip (V13 fix)."""
        B, _, T, V, M = x_s.shape
        coords = x_s[:, :2, :, :, :].mean(dim=4) if M > 1 else x_s[:, :2, :, :, 0]

        t_indices = [max(0, T // 4), T // 2, min(T - 1, 3 * T // 4)]
        all_part_coords = []
        for t_idx in t_indices:
            coords_t = coords[:, :, t_idx, :].clone()
            coords_t[:, 1, :] = -coords_t[:, 1, :]   # Y-flip
            frame_parts = [coords_t[:, :, g].mean(dim=2) for g in self.part_groups]
            all_part_coords.append(torch.stack(frame_parts, dim=1))  # (B, 5, 2)

        return torch.stack(all_part_coords, dim=0).mean(dim=0)       # (B, 5, 2)

    def _compute_skeleton_uncertainty(self, x_s):
        B, _, T, V, M = x_s.shape
        coords = x_s[:, :2, :, :, :].mean(dim=4) if M > 1 else x_s[:, :2, :, :, 0]
        velocity = coords[:, :, 1:, :] - coords[:, :, :-1, :]
        speed = velocity.pow(2).sum(dim=1).sqrt()
        mean_speed_sq = speed.mean(dim=1).pow(2)
        accel = velocity[:, :, 1:, :] - velocity[:, :, :-1, :]
        jitter = accel.pow(2).sum(dim=1).mean(dim=1)
        norm_jitter = jitter / (mean_speed_sq + 1e-4)
        return torch.cat(
            [norm_jitter[:, g].mean(dim=1, keepdim=True) for g in self.part_groups],
            dim=1
        )

    def compute_consistency_loss(self, confidence, fused_logits, rgb_only_logits, labels):
        with torch.no_grad():
            fused_correct = (fused_logits.argmax(1) == labels).float()
            rgb_correct   = (rgb_only_logits.argmax(1) == labels).float()
            target = torch.full_like(confidence.squeeze(-1), 0.5)
            target = torch.where((fused_correct * rgb_correct).bool(),
                                 torch.ones_like(target) * 0.9, target)
            target = torch.where((fused_correct * (1 - rgb_correct)).bool(),
                                 torch.ones_like(target), target)
            target = torch.where(((1 - fused_correct) * rgb_correct).bool(),
                                 torch.zeros_like(target), target)
            weight = torch.ones_like(target)
            weight = torch.where(((1 - fused_correct) * (1 - rgb_correct)).bool(),
                                 torch.ones_like(weight) * 0.3, weight)
        conf = confidence.squeeze(-1).clamp(1e-6, 1 - 1e-6)
        bce = -target * torch.log(conf) - (1 - target) * torch.log(1 - conf)
        return (bce * weight).mean()

    def forward(self, x_s, x_rgb, labels=None):
        with torch.no_grad():
            part_coords = self._extract_part_coords(x_s)
            _ = self._compute_skeleton_uncertainty(x_s)
            _, feature_s = self.ctrgcn.extract_feature(x_s)

        skel_grid = self._build_skel_grid(feature_s.detach())

        x = self.stem(x_rgb)
        x = self.layer1(x)
        x = self.layer2(x)

        if self.training:
            rgb_only_logits = self.rgb_only_head(x.detach())

        x_fused, confidence = self.cross_attn(
            x, skel_grid, part_coords, exp_type=self.exp_type
        )

        x_fused = self.layer3(x_fused)
        x_fused = self.layer4(x_fused)
        x_fused = self.avgpool(x_fused)
        x_fused = torch.flatten(x_fused, 1)
        output = self.fc(x_fused)

        if self.training and labels is not None:
            consistency_loss = self.compute_consistency_loss(
                confidence, output, rgb_only_logits, labels
            )
            return output, self.consistency_weight * consistency_loss

        return output
