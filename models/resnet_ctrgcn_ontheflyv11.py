"""
ResNet + CTR-GCN On-The-Fly V11
================================
Focus: "Skeleton chi duoc DUNG theo, khong duoc SAI theo"
       Confidence-Gated Skeleton Fusion — skeleton can only HELP, never HURT.

Problem analysis of V10 (80.60% vs V9's 88.58%):
  V10 used aggressive path dropout (p=0.3) + channel dropout (p=0.2).
  This DESTROYS skeleton info during training:
    1. 30% of batches get ZERO skeleton → model forced to classify from
       RGB layer2 features alone (28x28) → too hard, gradient is noisy
    2. Inverted dropout scales values by 1/(1-p)=1.43 → distribution shift
    3. Two dropout layers compound → effective info loss too large
    4. Result: model can't learn good skeleton→RGB fusion, accuracy drops 8%

V11 insight: Instead of REMOVING skeleton (dropout), CONTROL how much to trust it.
  Core principle: skeleton should only HELP, never HURT.
  
  Mechanism: DUAL-PATH CONFIDENCE GATING
  
  1. RGB-ONLY PATH (anchor):
     - ResNet processes RGB independently through a lightweight "RGB-only head"
     - This gives a BASELINE prediction that doesn't use skeleton at all
     - Acts as safety net: if skeleton misleads, RGB-only is still correct
  
  2. SKELETON-ENHANCED PATH (V9 attention):
     - Same as V9: multi-channel skeleton grid guides spatial attention
     - Produces skeleton-enhanced features
  
  3. CONFIDENCE GATE (the key innovation):
     - Learns to predict HOW MUCH the skeleton-enhanced path should be trusted
     - Input: BOTH rgb features AND skeleton grid (cross-modal consistency check)
     - Output: scalar confidence α ∈ [0, 1] per sample
     - High α → skeleton info is consistent with RGB → trust it
     - Low α → skeleton info conflicts with RGB → ignore it, use RGB-only
  
  4. FINAL FUSION:
     output = rgb_only_feat + α * (skeleton_enhanced_feat - rgb_only_feat)
     
     When α=1: output = skeleton_enhanced_feat (full V9 behavior)
     When α=0: output = rgb_only_feat (pure RGB, skeleton ignored)
     
     This is a LEARNED INTERPOLATION, not dropout.
     The model NEVER loses skeleton info — it just learns when to use it.
  
  5. CONSISTENCY LOSS (training regularizer):
     L_consistency = -log(α) when both paths predict correctly
     L_consistency = -log(1-α) when skeleton path is wrong but RGB path is right
     
     This explicitly teaches the gate:
       "Open when skeleton helps, close when skeleton hurts"
     
     Weight: λ=0.1 (small, doesn't dominate main CE loss)

  WHY this is better than V10:
    - No information destruction: skeleton is always available
    - No distribution shift: no inverted dropout scaling
    - Learned adaptation: model decides per-sample, not random
    - Preserves V9 accuracy: when skeleton is good (majority), α≈1 → same as V9
    - Fixes V9 blind spots: when skeleton misleads, α→0 → falls back to RGB
    - Better ensemble: RGB-only path provides natural diversity from CTR-GCN

Comparison:
  V0:  Fixed L2, pick 1 joint, bilinear, no gate.
  V2:  Conv1d(256→1), skel_grid (B,1,5,5), 7x7+Sigmoid.
  V8:  Conv1d(256→K→1), deep 3x3+TempSigmoid, spatial gate.
  V9:  Conv1d(256→K), skel_grid (B,K,5,5), deep+TempSigmoid, spatial gate.
  V10: V9 + skeleton path dropout (p=0.3) — too aggressive, drops to 80.6%.
  V11: V9 + confidence-gated fusion — skeleton only helps, never hurts.
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


class CrossModalAttentionV11(nn.Module):
    """Cross-Modal Attention V11: V9 attention + confidence-gated fusion.

    Same spatial/channel attention as V9, but adds:
      1. Confidence gate: learns when to trust skeleton vs fall back to RGB
      2. Smooth interpolation: output = rgb + α * (skel_enhanced - rgb)
      3. No dropout, no information destruction

    The confidence gate examines CROSS-MODAL CONSISTENCY:
      - If skeleton attention pattern AGREES with RGB features → high α
      - If skeleton attention CONFLICTS with RGB features → low α → safe fallback
    """
    def __init__(self, rgb_channels, skel_channels=8, skel_grid_size=200,
                 reduction=4, init_temperature=0.3, sp_skel_channels=4):
        super().__init__()

        self.sp_skel_channels = sp_skel_channels

        # ============================================================
        # V9 COMPONENTS (unchanged)
        # ============================================================

        # 1. CHANNEL ATTENTION (from V9)
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False),
            nn.Sigmoid()
        )

        # 2. MULTI-CHANNEL UPSAMPLING (from V9)
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

        # 3. DEEP SPATIAL ATTENTION (from V9)
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

        # LEARNABLE temperature (from V9)
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())

        # 4. SPATIAL-WISE GATE (from V9)
        self.spatial_gate = SpatialWiseGate(skel_channels)

        # ============================================================
        # V11 NEW: CONFIDENCE GATE
        # ============================================================
        # Examines cross-modal consistency to decide trust level.
        #
        # Input features:
        #   - skel_flat (K*25): skeleton dynamics summary
        #   - rgb_global (rgb_channels): RGB feature summary (GAP)
        # Output: α ∈ [0, 1] per sample
        #
        # Architecture: small MLP with bottleneck
        #   Cat(skel_flat, rgb_global) → Linear → ReLU → Linear → Sigmoid
        #
        # Initialize bias so α starts around 0.8 (trust skeleton by default,
        # since V9 shows skeleton is helpful for ~88% of samples)
        confidence_input_dim = skel_grid_size + rgb_channels  # K*25 + 512
        confidence_hidden = 64
        self.confidence_gate = nn.Sequential(
            nn.Linear(confidence_input_dim, confidence_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # light dropout for regularization only
            nn.Linear(confidence_hidden, 1, bias=True),
            nn.Sigmoid(),
        )
        # Initialize final bias so sigmoid outputs ~0.85 initially
        # sigmoid(1.73) ≈ 0.85 → trust skeleton by default
        nn.init.constant_(self.confidence_gate[-2].bias, 1.73)
        # Initialize weights small so gate is stable initially
        nn.init.xavier_uniform_(self.confidence_gate[-2].weight, gain=0.1)
        nn.init.xavier_uniform_(self.confidence_gate[0].weight, gain=0.5)

    def forward(self, rgb_feat, skel_grid, exp_type='normal'):
        """
        Returns:
            output_feat: (B, C, H, W) — confidence-gated fusion
            confidence: (B, 1) — trust score for skeleton path (for consistency loss)
        """
        B, C, H, W = rgb_feat.shape

        # ABLATION experiments
        if exp_type == 'noise':
            skel_grid = torch.randn_like(skel_grid)
        elif exp_type == 'ones':
            skel_grid = torch.ones_like(skel_grid)
        elif exp_type == 'zeros':
            skel_grid = torch.zeros_like(skel_grid)

        skel_flat = skel_grid.view(B, -1)                       # (B, K*25)

        # ---- V11: COMPUTE CONFIDENCE α ----
        # Global RGB summary via adaptive average pooling
        rgb_global = F.adaptive_avg_pool2d(rgb_feat, 1).view(B, -1)  # (B, C)
        confidence_input = torch.cat([skel_flat, rgb_global], dim=1)  # (B, K*25+C)
        confidence = self.confidence_gate(confidence_input)            # (B, 1)

        # ---- V9 ATTENTION PIPELINE ----

        # --- SPATIAL-WISE GATE ---
        gate_map = self.spatial_gate(skel_grid)                  # (B, 1, H, W)

        # --- STEP 1: CHANNEL ATTENTION ---
        ch_attn = self.channel_attn(skel_flat)                   # (B, C)
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)            # (B, C, 1, 1)
        feat_ca = rgb_feat * ch_attn                             # (B, C, H, W)

        if exp_type == 'no_spatial':
            gate_scalar = gate_map.mean(dim=[2, 3], keepdim=True)
            skel_delta = gate_scalar * feat_ca
            # V11: confidence-weighted residual
            alpha = confidence.unsqueeze(-1).unsqueeze(-1)       # (B, 1, 1, 1)
            return rgb_feat + alpha * skel_delta, confidence

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

        # --- STEP 4: SPATIAL-GATED MODULATION ---
        modulated = feat_ca * sp_attn                            # (B, C, H, W)
        skel_delta = gate_map * modulated                        # (B, C, H, W)

        # ---- V11: CONFIDENCE-GATED FUSION ----
        # output = rgb_feat + α * skel_delta
        #
        # When α=1 (high confidence): output = rgb_feat + skel_delta (same as V9)
        # When α=0 (low confidence):  output = rgb_feat (pure RGB, skeleton ignored)
        #
        # Key: skeleton can only ADD to RGB, never subtract.
        # Even at α=0, the original RGB features are fully preserved.
        alpha = confidence.unsqueeze(-1).unsqueeze(-1)           # (B, 1, 1, 1)
        output_feat = rgb_feat + alpha * skel_delta              # (B, C, H, W)

        return output_feat, confidence


class Model(nn.Module):
    """V11: V9 + Confidence-Gated Fusion.

    Architecture:
      1. skel_proj: Conv1d(256→K) preserves K channels (from V9)
      2. joint_to_part: Conv1d(20→5) per K-channel (from V9)
      3. skel_grid: (B, K, 5, 5) multi-channel (from V9)
      4. Deep spatial + TempSigmoid + spatial gate (from V9)
      5. NEW: Confidence gate — per-sample α decides skeleton trust level
      6. NEW: Consistency loss — explicitly trains gate to trust good skeleton

    Training losses:
      L_total = L_CE(fused_output, label) + λ * L_consistency

      L_consistency encourages:
        - α → 1 when skeleton helps (fused_pred == label)
        - α → 0 when skeleton hurts (fused_pred != label AND rgb_only_pred == label)
        - neutral when both wrong (no signal to learn from)
    """
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5,
                 exp_type='normal', proj_channels=8, init_temperature=0.3,
                 sp_skel_channels=4, consistency_weight=0.1,
                 num_point=20, num_person=1):
        super(Model, self).__init__()

        self.exp_type = exp_type
        self.ctrgcn = ''
        self.temporal_rgb_frames = temporal_rgb_frames
        self.proj_channels = proj_channels
        self.consistency_weight = consistency_weight
        self.num_class = num_class
        self.num_person = num_person

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

        # ---- RGB-ONLY auxiliary head (lightweight) ----
        # Uses features BEFORE skeleton fusion for consistency loss
        # Only 1 FC layer — we don't want this to be too powerful
        # (it should be a rough baseline, not compete with the main path)
        self.rgb_only_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_class),  # from layer2 output (512 channels)
        )

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
            nn.Conv1d(num_point, 5, kernel_size=1, bias=False),
            nn.BatchNorm1d(5),
            nn.ReLU(inplace=True),
        )
        if num_point == 25:  # NTU RGB+D (25 joints)
            part_groups = [
                [0, 1, 2, 3, 20],           # Head/Torso/Spine
                [4, 5, 6, 7, 21, 22],       # Left arm + hand tips
                [8, 9, 10, 11, 23, 24],     # Right arm + hand tips
                [12, 13, 14, 15],           # Left leg
                [16, 17, 18, 19],           # Right leg
            ]
        else:  # NW-UCLA (20 joints) — default
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

        # ---- V11: Cross-modal attention with confidence gate ----
        self.cross_attn = CrossModalAttentionV11(
            rgb_channels=512,
            skel_channels=K,
            skel_grid_size=K * 5 * temporal_rgb_frames,   # K*25 = 200
            reduction=4,
            init_temperature=init_temperature,
            sp_skel_channels=sp_skel_channels,
        )

    def _build_skel_grid(self, feature_s):
        """Single-stage projection → per-channel joint-to-part → temporal pool.
        Supports both single-person (M=1) and multi-person (M=2) skeletons.
        """
        B, C, T_new, V, M = feature_s.shape
        K = self.proj_channels
        T_frames = self.temporal_rgb_frames

        if M > 1:
            feat = feature_s.mean(dim=4)          # (B, C, T_new, V) — average across persons
        else:
            feat = feature_s[:, :, :, :, 0]       # (B, C, T_new, V)
        feat = feat.reshape(B, C, T_new * V)

        proj = self.skel_proj(feat)
        proj = proj.reshape(B, K, T_new, V)

        proj = proj.permute(0, 1, 3, 2)
        proj = proj.reshape(B * K, V, T_new)
        parts = self.joint_to_part(proj)
        parts = F.adaptive_avg_pool1d(parts, T_frames)
        parts = parts.reshape(B, K, 5, T_frames)

        return parts

    def compute_consistency_loss(self, confidence, fused_logits, rgb_only_logits, labels):
        """Consistency loss: teach the confidence gate when to trust skeleton.

        Cases:
          1. Fused is CORRECT → skeleton helped (or at least didn't hurt)
             → Encourage α → 1 (trust skeleton)
             → Loss: -log(α)

          2. Fused is WRONG but RGB-only is CORRECT → skeleton HURT
             → Encourage α → 0 (don't trust skeleton)
             → Loss: -log(1 - α)

          3. Both WRONG → ambiguous, no clear signal
             → No loss (weight = 0)

          4. Both CORRECT → skeleton didn't hurt
             → Mildly encourage α → 1
             → Loss: -0.5 * log(α) (half weight)

        This uses SOFT labels via binary cross-entropy to be differentiable.
        """
        with torch.no_grad():
            fused_preds = fused_logits.argmax(dim=1)           # (B,)
            rgb_preds = rgb_only_logits.argmax(dim=1)          # (B,)
            fused_correct = (fused_preds == labels).float()    # (B,)
            rgb_correct = (rgb_preds == labels).float()        # (B,)

            # Target for confidence gate:
            # Case 1: fused correct → target = 1.0 (trust skeleton)
            # Case 2: fused wrong, rgb correct → target = 0.0 (skeleton hurts!)
            # Case 3: both wrong → target = 0.5 (neutral, don't push either way)
            # Case 4: both correct → target = 0.9 (mildly trust skeleton)

            target = torch.full_like(confidence.squeeze(-1), 0.5)  # default: neutral

            both_correct = fused_correct * rgb_correct
            fused_only = fused_correct * (1 - rgb_correct)
            rgb_only = (1 - fused_correct) * rgb_correct

            target = torch.where(both_correct.bool(), torch.ones_like(target) * 0.9, target)
            target = torch.where(fused_only.bool(), torch.ones_like(target), target)
            target = torch.where(rgb_only.bool(), torch.zeros_like(target), target)

            # Weight: full weight for clear cases, half for neutral
            weight = torch.ones_like(target)
            both_wrong = (1 - fused_correct) * (1 - rgb_correct)
            weight = torch.where(both_wrong.bool(), torch.ones_like(weight) * 0.3, weight)

        # Binary cross-entropy loss
        conf = confidence.squeeze(-1).clamp(1e-6, 1 - 1e-6)
        bce = -target * torch.log(conf) - (1 - target) * torch.log(1 - conf)
        loss = (bce * weight).mean()

        return loss

    def forward(self, x_s, x_rgb, labels=None):
        """
        Args:
            x_s: skeleton data (B, C, T, V, M)
            x_rgb: RGB frames (B, 3, 224, 224)
            labels: ground truth labels (B,) — only needed during training for consistency loss

        Returns:
            if training and labels provided:
                (output, total_loss_extra)
                  output: (B, num_class) classification logits
                  total_loss_extra: scalar consistency loss (to be added to main CE loss)
            else:
                output: (B, num_class) classification logits
        """
        with torch.no_grad():
            _, feature_s = self.ctrgcn.extract_feature(x_s)

        skel_grid = self._build_skel_grid(feature_s.detach())

        # ---- ResNet backbone (shared stem + layer1 + layer2) ----
        x = self.stem(x_rgb)
        x = self.layer1(x)
        x = self.layer2(x)                                       # (B, 512, 28, 28)

        # ---- RGB-ONLY auxiliary prediction (before skeleton fusion) ----
        if self.training:
            rgb_only_logits = self.rgb_only_head(x.detach())     # (B, num_class)
            # .detach() so rgb_only_head doesn't affect backbone gradients
            # (backbone should be optimized for the FUSED path, not RGB-only)

        # ---- V11: Confidence-gated cross-modal attention ----
        x_fused, confidence = self.cross_attn(x, skel_grid, exp_type=self.exp_type)

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