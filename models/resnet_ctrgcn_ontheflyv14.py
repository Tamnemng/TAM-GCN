"""
ResNet + CTR-GCN On-The-Fly V14
================================
Inherits all 4 bug fixes from V13, adds 2 targeted fixes for
Doffing and Carry regression vs V11.

ROOT CAUSE of V13 Doffing/Carry regression vs V11:
  1. V13 dropped SpatialWiseGate — V11's transposed-conv upsampler maps
     skeleton dynamics directly to 28x28 spatial attention using ALL temporal
     frames. V13 replaced this with Gaussian heatmaps (coordinate-based),
     losing the temporal-aggregate spatial coverage that helps Doffing/Carry.
  2. V13 uses only middle frame (T//2) for Gaussian heatmap placement.
     Doffing is a dynamic action spanning the full sequence — middle frame
     may capture a non-representative arm pose. Same issue for Carry.

V14 FIX 1 — Restore SpatialWiseGate (from V11):
  Added back V11's transposed-conv spatial gate as a multiplicative factor
  on skel_delta. This provides a temporally-aggregated spatial attention path
  that is coordinate-free and captures skeleton dynamics across all frames.
  Result: skel_delta = feat_ca * sp_attn * gate_map  (vs V13: feat_ca * sp_attn)

V14 FIX 2 — Multi-frame coordinate extraction:
  Instead of T//2 only, sample 3 keyframes [T//4, T//2, 3*T//4] and average
  the coordinates. This gives a more representative body part center for
  dynamic actions (Doffing, Carry) where arm positions change throughout.

V14 keeps all V13 fixes unchanged:
  [V13 BUG 1] Y-axis flip + middle frame → now multi-frame
  [V13 BUG 2] Normalized jitter = accel_var / (speed² + ε)
  [V13 BUG 3] Sigmoid cross-attention (not softmax)
  [V13 BUG 4] Confidence gate uses only [skel_flat, rgb_global]

Comparison:
  V11: SpatialWiseGate + skel_upsample + temporal-sigmoid spatial attn. No coords.
  V13: Gaussian scatter + sigmoid cross-attn. Coords from middle frame only.
  V14: V13 + SpatialWiseGate restored + multi-frame coords (3 keyframes).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialWiseGate(nn.Module):
    """V11 transposed-conv spatial gate (restored in V14).

    Upsamples skeleton grid (B, K, 5, T) directly to (B, 1, H, W) via
    transposed convolutions — coordinate-free, uses all temporal dynamics.
    Provides complementary spatial coverage to Gaussian heatmaps.
    """
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
        # Initialize bias so gate starts permissive (~0.85)
        nn.init.constant_(self.gate_net[-2].bias, 0.85)

    def forward(self, skel_grid):
        return self.gate_net(skel_grid)  # (B, 1, H, W)


class CrossModalAttentionV14(nn.Module):
    """Cross-Modal Attention V14: V13 + SpatialWiseGate restored.

    Spatial attention paths:
      Path A: Heatmap-guided feature scatter (sharp + coarse Gaussians)
      Path B: Sigmoid cross-attention (coordinate-free)
      Path C: SpatialWiseGate (V11 transposed-conv, coordinate-free) — NEW in V14

    Final fusion:
      skel_delta = feat_ca * sp_attn * gate_map   (V13 had no gate_map)
      output = rgb_feat + alpha * skel_delta
    """
    def __init__(self, rgb_channels, skel_channels=8, skel_grid_size=200,
                 reduction=4, num_parts=5, sp_feat_channels=4,
                 init_sigma_sharp=3.0, init_sigma_coarse=8.0,
                 cross_attn_dim=16):
        super().__init__()

        self.num_parts = num_parts
        self.sp_feat_channels = sp_feat_channels
        self.cross_attn_dim = cross_attn_dim

        # ============================================================
        # 1. CHANNEL ATTENTION (same as V13)
        # ============================================================
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False),
            nn.Sigmoid()
        )

        # ============================================================
        # 2. HEATMAP-GUIDED FEATURE SCATTER (same as V13)
        # ============================================================
        self.part_feat_proj = nn.Sequential(
            nn.Conv1d(skel_channels, sp_feat_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(sp_feat_channels),
            nn.ReLU(inplace=True),
        )

        # Learnable coordinate adjustment (residual, init → identity)
        self.coord_adjust = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
        )
        nn.init.zeros_(self.coord_adjust[-1].weight)
        nn.init.zeros_(self.coord_adjust[-1].bias)

        # Learnable sigma for multi-scale Gaussians
        self.log_sigma_sharp = nn.Parameter(torch.tensor(init_sigma_sharp).log())
        self.log_sigma_coarse = nn.Parameter(torch.tensor(init_sigma_coarse).log())

        # ============================================================
        # 3. CROSS-ATTENTION (coordinate-free, sigmoid — V13 BUG 3 FIX)
        # ============================================================
        self.rgb_query_proj = nn.Conv2d(rgb_channels, cross_attn_dim, 1, bias=False)
        self.skel_key_proj = nn.Linear(skel_channels, cross_attn_dim, bias=False)
        self.skel_val_proj = nn.Linear(skel_channels, sp_feat_channels, bias=False)

        # ============================================================
        # 4. SPATIAL ATTENTION NETWORK (same as V13)
        # ============================================================
        # Input: rgb_max(1) + rgb_avg(1) + scatter_sharp(C_sp) +
        #        scatter_coarse(C_sp) + cross_attn(C_sp) = 2 + 3*C_sp
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

        # ============================================================
        # 5. SPATIAL-WISE GATE (V11 restored — V14 FIX 1)
        # ============================================================
        self.spatial_gate = SpatialWiseGate(skel_channels)

        # ============================================================
        # 6. CONFIDENCE GATE (same as V13 — no jitter, V13 BUG 4 FIX)
        # ============================================================
        confidence_input_dim = skel_grid_size + rgb_channels
        confidence_hidden = 64
        self.confidence_gate = nn.Sequential(
            nn.Linear(confidence_input_dim, confidence_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(confidence_hidden, 1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.confidence_gate[-2].bias, 1.73)
        nn.init.xavier_uniform_(self.confidence_gate[-2].weight, gain=0.1)
        nn.init.xavier_uniform_(self.confidence_gate[0].weight, gain=0.5)

    @staticmethod
    def _generate_gaussian_heatmaps(part_coords, H, W, sigma):
        """Generate 2D Gaussian heatmaps at body part positions.

        Args:
            part_coords: (B, P, 2) — normalized (x, y) in [-1, 1]
            H, W: spatial dimensions
            sigma: scalar — standard deviation in pixels

        Returns:
            heatmaps: (B, P, H, W)
        """
        B, P, _ = part_coords.shape
        device = part_coords.device

        cx = (part_coords[:, :, 0] + 1) / 2 * (W - 1)   # (B, P)
        cy = (part_coords[:, :, 1] + 1) / 2 * (H - 1)   # (B, P)

        yy = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)

        cx = cx.view(B, P, 1, 1)
        cy = cy.view(B, P, 1, 1)

        heatmaps = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        return heatmaps

    def forward(self, rgb_feat, skel_grid, part_coords, exp_type='normal'):
        """
        Args:
            rgb_feat:    (B, C, H, W) — ResNet layer2 features
            skel_grid:   (B, K, 5, T_frames) — CTR-GCN projected skeleton grid
            part_coords: (B, 5, 2) — body part (x, y) centers in [-1, 1],
                         Y-flipped, averaged over 3 keyframes (V14 FIX 2)
            exp_type:    ablation type

        Returns:
            output_feat: (B, C, H, W)
            confidence:  (B, 1)
        """
        B, C, H, W = rgb_feat.shape

        if exp_type == 'noise':
            skel_grid = torch.randn_like(skel_grid)
            part_coords = torch.rand_like(part_coords) * 2 - 1
        elif exp_type == 'ones':
            skel_grid = torch.ones_like(skel_grid)
        elif exp_type == 'zeros':
            skel_grid = torch.zeros_like(skel_grid)

        skel_flat = skel_grid.view(B, -1)  # (B, K*5*T)

        # ---- CONFIDENCE GATE (skel + RGB only, no jitter — V13 BUG 4 FIX) ----
        rgb_global = F.adaptive_avg_pool2d(rgb_feat, 1).view(B, -1)
        confidence_input = torch.cat([skel_flat, rgb_global], dim=1)
        confidence = self.confidence_gate(confidence_input)

        # ---- CHANNEL ATTENTION ----
        ch_attn = self.channel_attn(skel_flat)
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)
        feat_ca = rgb_feat * ch_attn

        if exp_type == 'no_spatial':
            alpha = confidence.unsqueeze(-1).unsqueeze(-1)
            return rgb_feat + alpha * feat_ca, confidence

        # ---- V14 FIX 1: SPATIAL-WISE GATE (restored from V11) ----
        gate_map = self.spatial_gate(skel_grid)              # (B, 1, H, W)

        # ---- PATH A: HEATMAP-GUIDED FEATURE SCATTER ----
        skel_parts = skel_grid.mean(dim=3)                          # (B, K, 5)
        skel_parts_proj = self.part_feat_proj(skel_parts)            # (B, C_sp, 5)

        adj_coords = part_coords + self.coord_adjust(part_coords)    # (B, 5, 2)
        adj_coords = adj_coords.clamp(-1, 1)

        sigma_sharp = self.log_sigma_sharp.exp()
        sigma_coarse = self.log_sigma_coarse.exp()
        hm_sharp = self._generate_gaussian_heatmaps(adj_coords, H, W, sigma_sharp)
        hm_coarse = self._generate_gaussian_heatmaps(adj_coords, H, W, sigma_coarse)

        scatter_sharp = torch.einsum('bcp,bphw->bchw', skel_parts_proj, hm_sharp)
        scatter_coarse = torch.einsum('bcp,bphw->bchw', skel_parts_proj, hm_coarse)

        # ---- PATH B: CROSS-ATTENTION (sigmoid — V13 BUG 3 FIX) ----
        rgb_q = self.rgb_query_proj(feat_ca)                         # (B, d, H, W)
        rgb_q = rgb_q.view(B, self.cross_attn_dim, -1).permute(0, 2, 1)  # (B, H*W, d)

        skel_parts_t = skel_parts.permute(0, 2, 1)                  # (B, 5, K)
        skel_k = self.skel_key_proj(skel_parts_t)                    # (B, 5, d)
        skel_v = self.skel_val_proj(skel_parts_t)                    # (B, 5, C_sp)

        attn_logits = torch.bmm(rgb_q, skel_k.permute(0, 2, 1))     # (B, H*W, 5)
        attn_logits = attn_logits / math.sqrt(self.cross_attn_dim)
        attn_weights = torch.sigmoid(attn_logits)                    # sigmoid, not softmax

        cross_out = torch.bmm(attn_weights, skel_v)                  # (B, H*W, C_sp)
        cross_out = cross_out.permute(0, 2, 1).view(B, self.sp_feat_channels, H, W)

        # ---- SPATIAL ATTENTION ----
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)

        sp_input = torch.cat([
            rgb_max, rgb_avg,
            scatter_sharp,
            scatter_coarse,
            cross_out,
        ], dim=1)

        sp_attn = torch.sigmoid(self.spatial_net(sp_input))          # (B, 1, H, W)

        # ---- FUSION (V14: add gate_map multiplicative gate) ----
        # V13: skel_delta = feat_ca * sp_attn
        # V14: skel_delta = feat_ca * sp_attn * gate_map
        #   gate_map (from SpatialWiseGate) provides temporal-aggregate
        #   spatial coverage — complementary to coordinate-based sp_attn.
        skel_delta = feat_ca * sp_attn * gate_map
        alpha = confidence.unsqueeze(-1).unsqueeze(-1)
        output_feat = rgb_feat + alpha * skel_delta

        return output_feat, confidence


class Model(nn.Module):
    """V14: V13 with 2 targeted fixes for Doffing/Carry regression.

    Changes from V13:
      FIX 1 — SpatialWiseGate restored (V11 transposed-conv upsampler).
               Applies as multiplicative gate on skel_delta.
               Provides temporal-aggregate coordinate-free spatial coverage.
      FIX 2 — Multi-frame coordinate extraction: averages part centers across
               3 keyframes [T//4, T//2, 3T//4] instead of just T//2.
               Better representative position for dynamic actions.

    All V13 bug fixes retained:
      [V13-1] Y-axis flip: Kinect Y-up → image Y-down
      [V13-2] Normalized jitter (accel / speed²)
      [V13-3] Sigmoid cross-attention (not softmax)
      [V13-4] Confidence gate: [skel_flat, rgb_global] only (no jitter)
    """
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5,
                 exp_type='normal', proj_channels=8, sp_feat_channels=4,
                 init_sigma_sharp=3.0, init_sigma_coarse=8.0,
                 cross_attn_dim=16, consistency_weight=0.1,
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

        # ---- Body part groups ----
        if num_point == 25:
            self.part_groups = [
                [0, 1, 2, 3, 20],
                [4, 5, 6, 7, 21, 22],
                [8, 9, 10, 11, 23, 24],
                [12, 13, 14, 15],
                [16, 17, 18, 19],
            ]
        else:
            self.part_groups = [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
                [16, 17, 18, 19],
            ]

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

        # ---- RGB-ONLY auxiliary head (for consistency loss) ----
        self.rgb_only_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_class),
        )

        # ---- Skeleton projection ----
        gcn_channels = 256
        K = proj_channels
        self.skel_proj = nn.Sequential(
            nn.Conv1d(gcn_channels, K, kernel_size=1, bias=False),
            nn.BatchNorm1d(K),
            nn.ReLU(inplace=True),
        )

        # ---- Learnable joint-to-part grouping ----
        self.joint_to_part = nn.Sequential(
            nn.Conv1d(num_point, 5, kernel_size=1, bias=False),
            nn.BatchNorm1d(5),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            self.joint_to_part[0].weight.zero_()
            for i, group in enumerate(self.part_groups):
                for j in group:
                    self.joint_to_part[0].weight[i, j, 0] = 1.0 / len(group)

        # ---- V14: Cross-modal attention ----
        self.cross_attn = CrossModalAttentionV14(
            rgb_channels=512,
            skel_channels=K,
            skel_grid_size=K * 5 * temporal_rgb_frames,
            reduction=4,
            num_parts=5,
            sp_feat_channels=sp_feat_channels,
            init_sigma_sharp=init_sigma_sharp,
            init_sigma_coarse=init_sigma_coarse,
            cross_attn_dim=cross_attn_dim,
        )

    def _build_skel_grid(self, feature_s):
        """Single-stage projection → per-channel joint-to-part → temporal pool."""
        B, C, T_new, V, M = feature_s.shape
        K = self.proj_channels
        T_frames = self.temporal_rgb_frames

        if M > 1:
            feat = feature_s.mean(dim=4)
        else:
            feat = feature_s[:, :, :, :, 0]
        feat = feat.reshape(B, C, T_new * V)

        proj = self.skel_proj(feat)
        proj = proj.reshape(B, K, T_new, V)

        proj = proj.permute(0, 1, 3, 2)
        proj = proj.reshape(B * K, V, T_new)
        parts = self.joint_to_part(proj)
        parts = F.adaptive_avg_pool1d(parts, T_frames)
        parts = parts.reshape(B, K, 5, T_frames)

        return parts

    def _extract_part_coords(self, x_s):
        """Extract body part center (x, y) from skeleton, multi-frame averaged.

        V14 FIX 2 vs V13:
          Instead of using only middle frame (T//2), sample 3 keyframes at
          T//4, T//2, and 3*T//4 and average the part center coordinates.
          This gives a more representative body part position for dynamic
          actions (Doffing, Carry) where arm positions vary throughout.

        All V13 fixes retained:
          - Y-axis flip: Kinect Y-up (positive = head) → image Y-down
          - Coordinates in normalized [-1, 1] space

        Args:
            x_s: (B, 3, T, V, M) — normalized skeleton coords in [-1, 1]
        Returns:
            part_coords: (B, 5, 2) — (x, y) in image-space [-1, 1]
        """
        B, _, T, V, M = x_s.shape

        if M > 1:
            coords = x_s[:, :2, :, :, :].mean(dim=4)   # (B, 2, T, V)
        else:
            coords = x_s[:, :2, :, :, 0]               # (B, 2, T, V)

        # Sample 3 keyframes for better temporal coverage
        t_indices = [max(0, T // 4), T // 2, min(T - 1, 3 * T // 4)]

        all_part_coords = []
        for t_idx in t_indices:
            coords_t = coords[:, :, t_idx, :].clone()   # (B, 2, V)

            # Y-axis flip: Kinect Y-up → image Y-down (V13 BUG 1 FIX)
            coords_t[:, 1, :] = -coords_t[:, 1, :]

            frame_parts = []
            for group in self.part_groups:
                part_center = coords_t[:, :, group].mean(dim=2)   # (B, 2)
                frame_parts.append(part_center)
            all_part_coords.append(torch.stack(frame_parts, dim=1))  # (B, 5, 2)

        # Average part centers across keyframes (V14 FIX 2)
        part_coords = torch.stack(all_part_coords, dim=0).mean(dim=0)  # (B, 5, 2)

        return part_coords

    def _compute_skeleton_uncertainty(self, x_s):
        """Normalized jitter as skeleton quality proxy (V13 BUG 2 FIX).

        normalized_jitter = acceleration_variance / (mean_speed² + ε)
        """
        B, _, T, V, M = x_s.shape

        if M > 1:
            coords = x_s[:, :2, :, :, :].mean(dim=4)
        else:
            coords = x_s[:, :2, :, :, 0]               # (B, 2, T, V)

        velocity = coords[:, :, 1:, :] - coords[:, :, :-1, :]      # (B, 2, T-1, V)
        speed = velocity.pow(2).sum(dim=1).sqrt()                    # (B, T-1, V)
        mean_speed_sq = speed.mean(dim=1).pow(2)                     # (B, V)

        accel = velocity[:, :, 1:, :] - velocity[:, :, :-1, :]      # (B, 2, T-2, V)
        jitter = accel.pow(2).sum(dim=1).mean(dim=1)                 # (B, V)

        norm_jitter = jitter / (mean_speed_sq + 1e-4)                # (B, V)

        uncertainty = []
        for group in self.part_groups:
            part_unc = norm_jitter[:, group].mean(dim=1, keepdim=True)
            uncertainty.append(part_unc)
        uncertainty = torch.cat(uncertainty, dim=1)                  # (B, 5)

        return uncertainty

    def compute_consistency_loss(self, confidence, fused_logits, rgb_only_logits, labels):
        """Consistency loss: teach confidence gate when to trust skeleton."""
        with torch.no_grad():
            fused_preds = fused_logits.argmax(dim=1)
            rgb_preds = rgb_only_logits.argmax(dim=1)
            fused_correct = (fused_preds == labels).float()
            rgb_correct = (rgb_preds == labels).float()

            target = torch.full_like(confidence.squeeze(-1), 0.5)

            both_correct = fused_correct * rgb_correct
            fused_only = fused_correct * (1 - rgb_correct)
            rgb_only_mask = (1 - fused_correct) * rgb_correct

            target = torch.where(both_correct.bool(), torch.ones_like(target) * 0.9, target)
            target = torch.where(fused_only.bool(), torch.ones_like(target), target)
            target = torch.where(rgb_only_mask.bool(), torch.zeros_like(target), target)

            weight = torch.ones_like(target)
            both_wrong = (1 - fused_correct) * (1 - rgb_correct)
            weight = torch.where(both_wrong.bool(), torch.ones_like(weight) * 0.3, weight)

        conf = confidence.squeeze(-1).clamp(1e-6, 1 - 1e-6)
        bce = -target * torch.log(conf) - (1 - target) * torch.log(1 - conf)
        loss = (bce * weight).mean()

        return loss

    def forward(self, x_s, x_rgb, labels=None):
        """
        Args:
            x_s:    skeleton data (B, 3, T, V, M)
            x_rgb:  RGB frames (B, 3, 224, 224)
            labels: ground truth (B,) — training only
        """
        with torch.no_grad():
            part_coords = self._extract_part_coords(x_s)
            _ = self._compute_skeleton_uncertainty(x_s)

        with torch.no_grad():
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
