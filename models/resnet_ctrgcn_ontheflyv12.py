"""
ResNet + CTR-GCN On-The-Fly V12
================================
Fix 3 fundamental issues of V11:

  1. SPATIAL FALLACY FIX → Gaussian Heatmaps from actual joint (x,y) coordinates
     V11 used ConvTranspose2d to upsample [BodyPart x Time] 5x5 matrix to 28x28,
     treating Part and Time axes as spatial Height and Width — mathematically nonsensical.
     V12 extracts real (x,y) joint coordinates from skeleton input, groups them into
     5 body parts, and generates Gaussian heatmaps directly on the 28x28 feature map.
     Attention is now ANCHORED to actual body positions.

  2. OVER-SHARPENING FIX → Multi-Scale Attention (Sharp + Coarse branches)
     V11 used a single learnable temperature to sharpen sigmoid, causing "local blindness"
     for whole-body actions like 'Sit down' (collapsed from 76.6% to 12.8% under noise).
     V12 uses TWO attention branches:
       - Sharp branch (small sigma): focuses on local body parts (good for hand gestures)
       - Coarse branch (large sigma): covers whole body (good for sit down, stand up)
     A per-sample gate learns to mix the two scales based on the action's spatial extent.

  3. BLIND CONFIDENCE FIX → Uncertainty-Aware Confidence Gate
     V11's MLP gate had no way to detect skeleton noise at inference (garbage in → garbage out).
     V12 computes temporal jitter (acceleration variance) from raw skeleton coordinates
     as a direct skeleton quality signal. High jitter = noisy/unreliable skeleton.
     This uncertainty score is fed into the confidence gate MLP alongside skeleton and
     RGB features, giving the gate explicit evidence to reduce α when skeleton is bad.

Architecture overview:
  Skeleton input (B, 3, T, V, M)
    ├→ CTR-GCN → features → skel_grid (B, K, 5, 5) → channel attention
    ├→ Raw (x,y) coords → body part positions → Gaussian Heatmaps on 28x28
    └→ Temporal jitter → uncertainty scores → confidence gate

  RGB input → ResNet stem → layer1 → layer2 → (B, 512, 28, 28)

  Cross-Modal Fusion:
    1. Channel attention from skel_grid (same as V11)
    2. Gaussian heatmaps at actual body positions (replaces ConvTranspose2d)
    3. Multi-scale spatial attention: sharp (local) + coarse (global)
    4. Confidence gate with skeleton uncertainty
    5. Residual fusion: output = rgb + α * skeleton_delta

Comparison:
  V0:  Fixed L2, pick 1 joint, bilinear, no gate.
  V2:  Conv1d(256→1), skel_grid (B,1,5,5), 7x7+Sigmoid.
  V9:  Conv1d(256→K), skel_grid (B,K,5,5), ConvTranspose2d, deep+TempSigmoid, spatial gate.
  V11: V9 + confidence-gated fusion (ConvTranspose2d spatial fallacy, single-scale, blind gate).
  V12: Gaussian heatmaps + multi-scale attention + uncertainty-aware confidence gate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CrossModalAttentionV12(nn.Module):
    """Cross-Modal Attention V12: Gaussian Heatmaps + Multi-Scale + Uncertainty Gate.

    Key differences from V11:
      1. No ConvTranspose2d — uses Gaussian heatmaps from real joint coordinates
      2. Two spatial attention branches (sharp + coarse) with per-sample mixing
      3. Confidence gate receives skeleton uncertainty (temporal jitter) as input
    """
    def __init__(self, rgb_channels, skel_channels=8, skel_grid_size=200,
                 reduction=4, num_parts=5, init_sigma_sharp=2.0, init_sigma_coarse=6.0):
        super().__init__()

        self.num_parts = num_parts

        # ============================================================
        # 1. CHANNEL ATTENTION (from V11/V9 — operates in feature space, no spatial issue)
        # ============================================================
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False),
            nn.Sigmoid()
        )

        # ============================================================
        # 2. GAUSSIAN HEATMAP PARAMETERS (replaces ConvTranspose2d)
        # ============================================================
        # Learnable sigma controls the spread of Gaussian blobs
        # Sharp: small sigma → focused attention on specific body parts
        # Coarse: large sigma → broad attention covering whole body
        self.log_sigma_sharp = nn.Parameter(torch.tensor(init_sigma_sharp).log())
        self.log_sigma_coarse = nn.Parameter(torch.tensor(init_sigma_coarse).log())

        # ============================================================
        # 3. MULTI-SCALE SPATIAL ATTENTION
        # ============================================================
        # Each branch: concat(rgb_max, rgb_avg, heatmaps) → conv → logits
        sp_input_channels = 2 + num_parts  # rgb_max + rgb_avg + 5 part heatmaps = 7
        sp_hidden = 8

        self.spatial_net_sharp = nn.Sequential(
            nn.Conv2d(sp_input_channels, sp_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sp_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(sp_hidden, sp_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sp_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(sp_hidden, 1, kernel_size=1, bias=True),
        )

        self.spatial_net_coarse = nn.Sequential(
            nn.Conv2d(sp_input_channels, sp_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sp_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(sp_hidden, sp_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sp_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(sp_hidden, 1, kernel_size=1, bias=True),
        )

        # Per-sample scale mixing gate: learns when to use sharp vs coarse
        # Input: skel_flat — skeleton dynamics tell us if action is local or global
        self.scale_gate = nn.Sequential(
            nn.Linear(skel_grid_size, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2, bias=True),
            # Output: 2 logits → softmax to get (w_sharp, w_coarse)
        )
        # Initialize to equal mixing (both logits = 0 → softmax = [0.5, 0.5])
        nn.init.zeros_(self.scale_gate[-1].weight)
        nn.init.zeros_(self.scale_gate[-1].bias)

        # ============================================================
        # 4. UNCERTAINTY-AWARE CONFIDENCE GATE
        # ============================================================
        # Input: skel_flat (K*25) + rgb_global (512) + uncertainty (num_parts)
        # The uncertainty signal gives the gate DIRECT evidence of skeleton quality
        confidence_input_dim = skel_grid_size + rgb_channels + num_parts
        confidence_hidden = 64
        self.confidence_gate = nn.Sequential(
            nn.Linear(confidence_input_dim, confidence_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(confidence_hidden, 1, bias=True),
            nn.Sigmoid(),
        )
        # Initialize to trust skeleton by default: sigmoid(1.73) ≈ 0.85
        nn.init.constant_(self.confidence_gate[-2].bias, 1.73)
        nn.init.xavier_uniform_(self.confidence_gate[-2].weight, gain=0.1)
        nn.init.xavier_uniform_(self.confidence_gate[0].weight, gain=0.5)

    @staticmethod
    def _generate_gaussian_heatmaps(part_coords, H, W, sigma):
        """Generate 2D Gaussian heatmaps at body part positions.

        Args:
            part_coords: (B, P, 2) — normalized (x, y) in [-1, 1]
            H, W: spatial dimensions of the output heatmap
            sigma: scalar or tensor — standard deviation of the Gaussian

        Returns:
            heatmaps: (B, P, H, W) — values in [0, 1]
        """
        B, P, _ = part_coords.shape
        device = part_coords.device

        # Map from [-1, 1] to [0, H-1] and [0, W-1]
        cx = (part_coords[:, :, 0] + 1) / 2 * (W - 1)   # (B, P)
        cy = (part_coords[:, :, 1] + 1) / 2 * (H - 1)   # (B, P)

        # Create coordinate grids — efficient broadcasting
        yy = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)

        cx = cx.view(B, P, 1, 1)
        cy = cy.view(B, P, 1, 1)

        # 2D Gaussian: exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
        heatmaps = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

        return heatmaps  # (B, P, H, W)

    def forward(self, rgb_feat, skel_grid, part_coords, uncertainty, exp_type='normal'):
        """
        Args:
            rgb_feat: (B, C, H, W) — ResNet layer2 features
            skel_grid: (B, K, 5, 5) — CTR-GCN projected skeleton grid
            part_coords: (B, 5, 2) — body part (x, y) centers in [-1, 1]
            uncertainty: (B, 5) — per-part temporal jitter (skeleton quality signal)
            exp_type: ablation experiment type

        Returns:
            output_feat: (B, C, H, W) — confidence-gated fusion result
            confidence: (B, 1) — trust score for skeleton path
        """
        B, C, H, W = rgb_feat.shape

        # ABLATION experiments
        if exp_type == 'noise':
            skel_grid = torch.randn_like(skel_grid)
            # Also randomize part coords and uncertainty
            part_coords = torch.rand_like(part_coords) * 2 - 1
            uncertainty = torch.rand_like(uncertainty)
        elif exp_type == 'ones':
            skel_grid = torch.ones_like(skel_grid)
        elif exp_type == 'zeros':
            skel_grid = torch.zeros_like(skel_grid)

        skel_flat = skel_grid.view(B, -1)  # (B, K*25)

        # ---- CONFIDENCE GATE with uncertainty ----
        rgb_global = F.adaptive_avg_pool2d(rgb_feat, 1).view(B, -1)  # (B, C)
        # Normalize uncertainty to [0, 1] range per-sample for stable MLP input
        unc_norm = uncertainty / (uncertainty.max(dim=1, keepdim=True)[0] + 1e-8)
        confidence_input = torch.cat([skel_flat, rgb_global, unc_norm], dim=1)
        confidence = self.confidence_gate(confidence_input)  # (B, 1)

        # ---- CHANNEL ATTENTION (from skel_grid features) ----
        ch_attn = self.channel_attn(skel_flat)                     # (B, C)
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)              # (B, C, 1, 1)
        feat_ca = rgb_feat * ch_attn                               # (B, C, H, W)

        if exp_type == 'no_spatial':
            # Ablation: skip spatial attention
            alpha = confidence.unsqueeze(-1).unsqueeze(-1)
            return rgb_feat + alpha * feat_ca, confidence

        # ---- GAUSSIAN HEATMAPS at actual body positions ----
        sigma_sharp = self.log_sigma_sharp.exp()
        sigma_coarse = self.log_sigma_coarse.exp()

        heatmaps_sharp = self._generate_gaussian_heatmaps(
            part_coords, H, W, sigma_sharp)      # (B, 5, H, W)
        heatmaps_coarse = self._generate_gaussian_heatmaps(
            part_coords, H, W, sigma_coarse)      # (B, 5, H, W)

        # RGB spatial cues
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]      # (B, 1, H, W)
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)         # (B, 1, H, W)

        # ---- MULTI-SCALE SPATIAL ATTENTION ----
        # Sharp branch: local attention for fine-grained actions
        sp_input_sharp = torch.cat([rgb_max, rgb_avg, heatmaps_sharp], dim=1)
        logits_sharp = self.spatial_net_sharp(sp_input_sharp)       # (B, 1, H, W)
        attn_sharp = torch.sigmoid(logits_sharp)                    # (B, 1, H, W)

        # Coarse branch: global attention for whole-body actions
        sp_input_coarse = torch.cat([rgb_max, rgb_avg, heatmaps_coarse], dim=1)
        logits_coarse = self.spatial_net_coarse(sp_input_coarse)    # (B, 1, H, W)
        attn_coarse = torch.sigmoid(logits_coarse)                  # (B, 1, H, W)

        # Per-sample scale mixing: skeleton dynamics determine sharp vs coarse
        scale_logits = self.scale_gate(skel_flat)                   # (B, 2)
        scale_weights = F.softmax(scale_logits, dim=1)              # (B, 2)
        w_sharp = scale_weights[:, 0:1].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        w_coarse = scale_weights[:, 1:2].unsqueeze(-1).unsqueeze(-1) # (B, 1, 1, 1)

        sp_attn = w_sharp * attn_sharp + w_coarse * attn_coarse     # (B, 1, H, W)

        # ---- SPATIAL-MODULATED FUSION ----
        skel_delta = feat_ca * sp_attn                              # (B, C, H, W)

        # ---- CONFIDENCE-GATED OUTPUT ----
        alpha = confidence.unsqueeze(-1).unsqueeze(-1)              # (B, 1, 1, 1)
        output_feat = rgb_feat + alpha * skel_delta                 # (B, C, H, W)

        return output_feat, confidence


class Model(nn.Module):
    """V12: Gaussian Heatmaps + Multi-Scale Attention + Uncertainty-Aware Confidence Gate.

    Architecture:
      1. skel_proj: Conv1d(256→K) preserves K channels (from V9/V11)
      2. joint_to_part: Conv1d(20→5) per K-channel (from V9/V11)
      3. skel_grid: (B, K, 5, 5) multi-channel (for channel attention only)
      4. NEW: Gaussian heatmaps from raw (x,y) coordinates (replaces ConvTranspose2d)
      5. NEW: Multi-scale spatial attention — sharp + coarse branches
      6. NEW: Uncertainty-aware confidence gate — temporal jitter as quality signal
      7. Consistency loss (from V11) — explicitly trains gate

    Training losses:
      L_total = L_CE(fused_output, label) + λ * L_consistency
    """
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5,
                 exp_type='normal', proj_channels=8, init_sigma_sharp=2.0,
                 init_sigma_coarse=6.0, consistency_weight=0.1,
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

        # ---- Body part groups (for coordinate extraction and joint-to-part) ----
        if num_point == 25:  # NTU RGB+D
            self.part_groups = [
                [0, 1, 2, 3, 20],           # Head/Torso/Spine
                [4, 5, 6, 7, 21, 22],       # Left arm + hand tips
                [8, 9, 10, 11, 23, 24],     # Right arm + hand tips
                [12, 13, 14, 15],           # Left leg
                [16, 17, 18, 19],           # Right leg
            ]
        else:  # NW-UCLA (20 joints)
            self.part_groups = [
                [0, 1, 2, 3],       # Head/Torso
                [4, 5, 6, 7],       # Left arm
                [8, 9, 10, 11],     # Right arm
                [12, 13, 14, 15],   # Left leg
                [16, 17, 18, 19],   # Right leg
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

        # ---- RGB-ONLY auxiliary head (for consistency loss, same as V11) ----
        self.rgb_only_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_class),
        )

        # ---- Single-stage projection preserving K channels (from V9/V11) ----
        gcn_channels = 256
        K = proj_channels
        self.skel_proj = nn.Sequential(
            nn.Conv1d(gcn_channels, K, kernel_size=1, bias=False),
            nn.BatchNorm1d(K),
            nn.ReLU(inplace=True),
        )

        # ---- Learnable joint-to-part grouping (from V9/V11) ----
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

        # ---- V12: Cross-modal attention with Gaussian heatmaps ----
        self.cross_attn = CrossModalAttentionV12(
            rgb_channels=512,
            skel_channels=K,
            skel_grid_size=K * 5 * temporal_rgb_frames,   # K*25
            reduction=4,
            num_parts=5,
            init_sigma_sharp=init_sigma_sharp,
            init_sigma_coarse=init_sigma_coarse,
        )

    def _build_skel_grid(self, feature_s):
        """Single-stage projection → per-channel joint-to-part → temporal pool.
        Same as V11 — used for channel attention (feature-space, not spatial).
        """
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
        """Extract body part center (x, y) coordinates from raw skeleton.

        Args:
            x_s: (B, 3, T, V, M) — raw skeleton coordinates in [-1, 1]

        Returns:
            part_coords: (B, 5, 2) — (x, y) center of each body part, in [-1, 1]
        """
        B = x_s.shape[0]

        # Take (x, y) coordinates, ignore z
        if x_s.shape[4] > 1:
            coords = x_s[:, :2, :, :, :].mean(dim=4)  # (B, 2, T, V)
        else:
            coords = x_s[:, :2, :, :, 0]               # (B, 2, T, V)

        # Average across time for stable position estimates
        coords = coords.mean(dim=2)                     # (B, 2, V)

        # Group by body parts → compute center of each part
        part_coords = []
        for group in self.part_groups:
            part_center = coords[:, :, group].mean(dim=2)  # (B, 2)
            part_coords.append(part_center)
        part_coords = torch.stack(part_coords, dim=1)      # (B, 5, 2)

        return part_coords

    def _compute_skeleton_uncertainty(self, x_s):
        """Compute temporal jitter as skeleton quality proxy.

        Jitter = mean squared acceleration of joints.
        High jitter means the skeleton detector was noisy/unstable.
        This gives the confidence gate DIRECT evidence of skeleton quality,
        unlike V11 which had no way to detect noise at inference.

        Args:
            x_s: (B, 3, T, V, M) — raw skeleton coordinates

        Returns:
            uncertainty: (B, 5) — per-body-part jitter scores
        """
        B = x_s.shape[0]

        if x_s.shape[4] > 1:
            coords = x_s[:, :2, :, :, :].mean(dim=4)  # (B, 2, T, V)
        else:
            coords = x_s[:, :2, :, :, 0]               # (B, 2, T, V)

        # Velocity: 1st derivative of position
        velocity = coords[:, :, 1:, :] - coords[:, :, :-1, :]    # (B, 2, T-1, V)

        # Acceleration: 2nd derivative — measures jitter/instability
        accel = velocity[:, :, 1:, :] - velocity[:, :, :-1, :]   # (B, 2, T-2, V)

        # Per-joint jitter: mean squared acceleration over time
        jitter = accel.pow(2).sum(dim=1).mean(dim=1)              # (B, V)

        # Group by body parts
        uncertainty = []
        for group in self.part_groups:
            part_jitter = jitter[:, group].mean(dim=1, keepdim=True)  # (B, 1)
            uncertainty.append(part_jitter)
        uncertainty = torch.cat(uncertainty, dim=1)                    # (B, 5)

        return uncertainty

    def compute_consistency_loss(self, confidence, fused_logits, rgb_only_logits, labels):
        """Consistency loss: teach the confidence gate when to trust skeleton.
        Same as V11 — the uncertainty input to the gate makes this even more effective.

        Cases:
          1. Fused CORRECT → target = 1.0 (trust skeleton)
          2. Fused WRONG but RGB-only CORRECT → target = 0.0 (skeleton HURT)
          3. Both WRONG → target = 0.5 (neutral)
          4. Both CORRECT → target = 0.9 (mildly trust skeleton)
        """
        with torch.no_grad():
            fused_preds = fused_logits.argmax(dim=1)
            rgb_preds = rgb_only_logits.argmax(dim=1)
            fused_correct = (fused_preds == labels).float()
            rgb_correct = (rgb_preds == labels).float()

            target = torch.full_like(confidence.squeeze(-1), 0.5)

            both_correct = fused_correct * rgb_correct
            fused_only = fused_correct * (1 - rgb_correct)
            rgb_only = (1 - fused_correct) * rgb_correct

            target = torch.where(both_correct.bool(), torch.ones_like(target) * 0.9, target)
            target = torch.where(fused_only.bool(), torch.ones_like(target), target)
            target = torch.where(rgb_only.bool(), torch.zeros_like(target), target)

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
            x_s: skeleton data (B, C, T, V, M) — raw coordinates in [-1, 1]
            x_rgb: RGB frames (B, 3, 224, 224)
            labels: ground truth labels (B,) — only needed during training

        Returns:
            if training and labels provided:
                (output, consistency_loss)
            else:
                output: (B, num_class) classification logits
        """
        # ---- Extract raw skeleton info BEFORE CTR-GCN ----
        with torch.no_grad():
            part_coords = self._extract_part_coords(x_s)           # (B, 5, 2)
            uncertainty = self._compute_skeleton_uncertainty(x_s)   # (B, 5)

        # ---- CTR-GCN feature extraction (frozen) ----
        with torch.no_grad():
            _, feature_s = self.ctrgcn.extract_feature(x_s)

        skel_grid = self._build_skel_grid(feature_s.detach())      # (B, K, 5, 5)

        # ---- ResNet backbone: stem → layer1 → layer2 ----
        x = self.stem(x_rgb)
        x = self.layer1(x)
        x = self.layer2(x)                                         # (B, 512, 28, 28)

        # ---- RGB-ONLY auxiliary prediction (for consistency loss) ----
        if self.training:
            rgb_only_logits = self.rgb_only_head(x.detach())

        # ---- V12: Cross-modal attention with Gaussian heatmaps ----
        x_fused, confidence = self.cross_attn(
            x, skel_grid, part_coords, uncertainty, exp_type=self.exp_type
        )

        # ---- Continue ResNet backbone ----
        x_fused = self.layer3(x_fused)
        x_fused = self.layer4(x_fused)
        x_fused = self.avgpool(x_fused)
        x_fused = torch.flatten(x_fused, 1)
        output = self.fc(x_fused)

        # ---- Consistency loss (training only) ----
        if self.training and labels is not None:
            consistency_loss = self.compute_consistency_loss(
                confidence, output, rgb_only_logits, labels
            )
            return output, self.consistency_weight * consistency_loss

        return output
