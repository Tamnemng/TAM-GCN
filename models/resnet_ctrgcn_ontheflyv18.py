"""
ResNet + CTR-GCN On-The-Fly V18
================================
Simplified from V17: remove spatial attention entirely (was near-uniform).

Changes from V17:
  - REMOVED: Gaussian scatter, coord_adjust, part_coords, _extract_part_coords
  - REMOVED: SpatialWiseGate (ConvTranspose spatial fallacy)
  - REMOVED: Dual-branch spatial net (sharp/coarse), branch_mixer
  - KEPT:    Channel Attention (skeleton guides which RGB channels matter)
  - KEPT:    Confidence Gate + norm_jitter (detects skeleton corruption)
  - KEPT:    rgb_only_head + consistency_loss (trains confidence gate)

Rationale:
  V17 spatial attention had entropy 6.631/6.664 (99.5% uniform) — effectively
  no spatial selectivity. Gaussian scatter used Kinect (x,y) coordinates that
  don't match the fixed ST-ROI grid layout. Removing it simplifies the model
  without losing meaningful signal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CrossModalAttentionV18(nn.Module):
    """Cross-Modal Attention V18 — Channel Attention + Confidence Gate only."""

    def __init__(self, rgb_channels, skel_grid_size=200,
                 reduction=4, num_parts=5):
        super().__init__()

        # ── CHANNEL ATTENTION ──────────────────────────────────────────────
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False),
            nn.Sigmoid(),
        )

        # ── CONFIDENCE GATE with jitter ───────────────────────────────────
        conf_in = skel_grid_size + rgb_channels + num_parts  # 200+512+5=717
        self.confidence_gate = nn.Sequential(
            nn.Linear(conf_in, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.confidence_gate[-2].bias, 0.5)
        nn.init.xavier_uniform_(self.confidence_gate[-2].weight, gain=0.1)
        nn.init.xavier_uniform_(self.confidence_gate[0].weight, gain=0.5)

    def forward(self, rgb_feat, skel_grid, norm_jitter, exp_type='normal'):
        B, C, H, W = rgb_feat.shape

        if exp_type == 'noise':
            skel_grid = torch.randn_like(skel_grid)
        elif exp_type == 'ones':
            skel_grid = torch.ones_like(skel_grid)
        elif exp_type == 'zeros':
            skel_grid = torch.zeros_like(skel_grid)

        skel_flat = skel_grid.view(B, -1)

        # ── CONFIDENCE GATE ──────────────────────────────────────────────
        rgb_global = F.adaptive_avg_pool2d(rgb_feat, 1).view(B, -1)
        confidence = self.confidence_gate(
            torch.cat([skel_flat, rgb_global, norm_jitter], dim=1)
        )

        # ── CHANNEL ATTENTION ────────────────────────────────────────────
        ch_attn = self.channel_attn(skel_flat).unsqueeze(-1).unsqueeze(-1)
        feat_ca = rgb_feat * ch_attn

        # ── FUSION ───────────────────────────────────────────────────────
        alpha = confidence.unsqueeze(-1).unsqueeze(-1)
        output_feat = rgb_feat + alpha * feat_ca
        return output_feat, confidence


class Model(nn.Module):
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5,
                 exp_type='normal', proj_channels=8,
                 consistency_weight=0.1, num_point=20, num_person=1,
                 **kwargs):
        super().__init__()
        self.exp_type            = exp_type
        self.ctrgcn              = ''
        self.temporal_rgb_frames = temporal_rgb_frames
        self.proj_channels       = proj_channels
        self.consistency_weight  = consistency_weight
        self.num_class           = num_class
        self.num_person          = num_person
        self.num_point           = num_point

        if num_point == 25:
            self.part_groups = [
                [0,1,2,3,20], [4,5,6,7,21,22],
                [8,9,10,11,23,24], [12,13,14,15], [16,17,18,19],
            ]
        else:
            self.part_groups = [
                [0,1,2,3], [4,5,6,7],
                [8,9,10,11], [12,13,14,15], [16,17,18,19],
            ]

        # ── RESNET-50 BACKBONE ───────────────────────────────────────────
        resnet       = models.resnet50(pretrained=pretrained)
        self.stem    = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc      = nn.Linear(resnet.fc.in_features, num_class)

        # ── RGB-ONLY HEAD (for consistency loss) ─────────────────────────
        self.rgb_only_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(512, num_class),
        )

        # ── SKELETON PROJECTION ──────────────────────────────────────────
        K = proj_channels
        self.skel_proj = nn.Sequential(
            nn.Conv1d(256, K, kernel_size=1, bias=False),
            nn.BatchNorm1d(K), nn.ReLU(inplace=True),
        )
        self.joint_to_part = nn.Sequential(
            nn.Conv1d(num_point, 5, kernel_size=1, bias=False),
            nn.BatchNorm1d(5), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            self.joint_to_part[0].weight.zero_()
            for i, grp in enumerate(self.part_groups):
                for j in grp:
                    self.joint_to_part[0].weight[i, j, 0] = 1.0 / len(grp)

        # ── CROSS-MODAL ATTENTION ────────────────────────────────────────
        self.cross_attn = CrossModalAttentionV18(
            rgb_channels=512,
            skel_grid_size=K * 5 * temporal_rgb_frames,
            reduction=4, num_parts=5,
        )

    def _build_skel_grid(self, feature_s):
        B, C, T_new, V, M = feature_s.shape
        K, T_fr = self.proj_channels, self.temporal_rgb_frames
        feat  = feature_s.mean(dim=4) if M > 1 else feature_s[:,:,:,:,0]
        feat  = feat.reshape(B, C, T_new * V)
        proj  = self.skel_proj(feat).reshape(B, K, T_new, V)
        proj  = proj.permute(0,1,3,2).reshape(B*K, V, T_new)
        parts = self.joint_to_part(proj)
        parts = F.adaptive_avg_pool1d(parts, T_fr)
        return parts.reshape(B, K, 5, T_fr)

    def _compute_skeleton_uncertainty(self, x_s):
        """Normalized jitter (B, 5) — proxy for skeleton detection quality."""
        B, _, T, V, M = x_s.shape
        c = x_s[:,:2,:,:,:].mean(dim=4) if M > 1 else x_s[:,:2,:,:,0]

        c_flat = c.reshape(B, 2, -1)
        c_min  = c_flat.min(dim=2, keepdim=True)[0].unsqueeze(2)
        c_max  = c_flat.max(dim=2, keepdim=True)[0].unsqueeze(2)
        c = 2.0 * (c - c_min) / (c_max - c_min + 1e-6) - 1.0

        vel        = c[:,:,1:,:] - c[:,:,:-1,:]
        speed      = vel.pow(2).sum(dim=1).sqrt()
        accel      = vel[:,:,1:,:] - vel[:,:,:-1,:]
        jitter     = accel.pow(2).sum(dim=1).mean(dim=1)
        mean_sp_sq = speed.mean(dim=1).pow(2)
        norm_j     = jitter / (mean_sp_sq + 1e-4)
        return torch.cat(
            [norm_j[:,g].mean(dim=1, keepdim=True) for g in self.part_groups],
            dim=1,
        )

    def compute_consistency_loss(self, confidence, fused_logits, rgb_only_logits, labels):
        with torch.no_grad():
            fc = (fused_logits.argmax(1) == labels).float()
            rc = (rgb_only_logits.argmax(1) == labels).float()
            target = torch.full_like(confidence.squeeze(-1), 0.5)
            target = torch.where((fc*rc).bool(),         torch.ones_like(target)*0.9, target)
            target = torch.where((fc*(1-rc)).bool(),     torch.ones_like(target),     target)
            target = torch.where(((1-fc)*rc).bool(),     torch.zeros_like(target),    target)
            weight = torch.ones_like(target)
            weight = torch.where(((1-fc)*(1-rc)).bool(), torch.ones_like(weight)*0.3, weight)
        conf = confidence.squeeze(-1).clamp(1e-6, 1-1e-6)
        bce  = -target*torch.log(conf) - (1-target)*torch.log(1-conf)
        return (bce * weight).mean()

    def forward(self, x_s, x_rgb, labels=None):
        with torch.no_grad():
            norm_jitter = self._compute_skeleton_uncertainty(x_s)
            _, feature_s = self.ctrgcn.extract_feature(x_s)

        skel_grid = self._build_skel_grid(feature_s.detach())

        x = self.stem(x_rgb)
        x = self.layer1(x)
        x = self.layer2(x)

        if self.training:
            rgb_only_logits = self.rgb_only_head(x.detach())

        x_fused, confidence = self.cross_attn(
            x, skel_grid, norm_jitter, exp_type=self.exp_type
        )

        x_fused = self.layer3(x_fused)
        x_fused = self.layer4(x_fused)
        x_fused = self.avgpool(x_fused)
        x_fused = torch.flatten(x_fused, 1)
        output  = self.fc(x_fused)

        if self.training and labels is not None:
            consistency_loss = self.compute_consistency_loss(
                confidence, output, rgb_only_logits, labels
            )
            return output, self.consistency_weight * consistency_loss

        return output
