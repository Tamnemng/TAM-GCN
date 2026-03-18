"""
ResNet + CTR-GCN On-The-Fly V17
================================
Implements 3 targeted fixes addressing theoretical weaknesses of V11-V16.

Fix 1 — Spatial Fallacy (ConvTranspose2d on Part x Time axes):
  V11-V16 used skel_upsample: ConvTranspose2d on skel_grid (B, K, 5_parts, 5_time)
  to produce (B, C, 28, 28). This treats Part and Time axes as spatial (H, W),
  semantically wrong — the model rote-learns (Part, Time) -> (H, W) mapping.
  Under noise, this misfires and misdirects attention to wrong image regions.
  FIX: Remove skel_upsample entirely. Use ONLY Gaussian heatmaps anchored to
       real (x, y) joint coordinates. Attention is physically grounded.
  NOTE: SpatialWiseGate still uses ConvTranspose, but it is a scalar magnitude
       gate (0~1), not an attention direction — so no spatial fallacy.

Fix 2 — Over-sharpening / Local Blindness (single Temperature T):
  V11/V16: sp_attn = sigmoid(logits / T), one T for all actions.
  Small T -> sharp attention -> "Sit down" collapses (12.8% under noise)
  because model focuses on small wrong spot instead of whole body.
  FIX: Two independent spatial_net branches with separate temperatures:
    Branch SHARP  (T_s init=0.3): local focus — hands/feet (Throw, Pick up)
    Branch COARSE (T_c init=1.5): global focus — whole body (Sit down, Walk)
    sp_attn = w * sp_sharp + (1-w) * sp_coarse
    w = sigmoid(Linear(skel_flat)) — learned per-sample branch weight.

Fix 3 — Confidence Gate blind to skeleton corruption (garbage in, garbage out):
  V11-V16: confidence_gate input = [skel_flat, rgb_global].
  When skeleton = noise, skel_flat is garbage -> gate outputs 0.97 (high confidence).
  FIX: Add norm_jitter (B, 5) to confidence gate input.
  norm_jitter = per-part |acceleration| / (speed^2 + eps) — high when joints
  jump discontinuously (noise/occlusion). Computable at inference without labels.
  Previously computed in _compute_skeleton_uncertainty() but discarded (_ =).
  Now actually connected to the gate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialWiseGate(nn.Module):
    """Transposed-conv spatial gate.
    Acts as magnitude modulation (0~1 scalar map), not attention direction.
    No spatial fallacy: it gates HOW MUCH to attend, not WHERE.
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
        nn.init.constant_(self.gate_net[-2].bias, 0.85)

    def forward(self, skel_grid):
        return self.gate_net(skel_grid)  # (B, 1, H, W)


def _make_spatial_net(in_ch, hidden=8):
    """Small spatial attention CNN — same structure as V11."""
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(hidden),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(hidden),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
    )


class CrossModalAttentionV17(nn.Module):
    """Cross-Modal Attention V17.

    Spatial attention: pure Gaussian scatter, dual-branch (sharp + coarse).
    Confidence gate: [skel_flat, rgb_global, norm_jitter] — jitter detects corruption.
    """
    def __init__(self, rgb_channels, skel_channels=8, skel_grid_size=200,
                 reduction=4, num_parts=5, sp_feat_channels=4,
                 init_sigma_sharp=3.0, init_sigma_coarse=8.0,
                 init_temp_sharp=0.3, init_temp_coarse=1.5):
        super().__init__()
        self.sp_feat_channels = sp_feat_channels

        # ── CHANNEL ATTENTION ──────────────────────────────────────────────
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels, bias=False),
            nn.Sigmoid(),
        )

        # ── GAUSSIAN SCATTER (Fix 1: only spatial path, no ConvTranspose) ─
        self.part_feat_proj = nn.Sequential(
            nn.Conv1d(skel_channels, sp_feat_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(sp_feat_channels),
            nn.ReLU(inplace=True),
        )
        self.coord_adjust = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(inplace=True), nn.Linear(16, 2),
        )
        nn.init.zeros_(self.coord_adjust[-1].weight)
        nn.init.zeros_(self.coord_adjust[-1].bias)

        self.log_sigma_sharp  = nn.Parameter(torch.tensor(init_sigma_sharp).log())
        self.log_sigma_coarse = nn.Parameter(torch.tensor(init_sigma_coarse).log())

        # ── BRANCH SHARP: local focus (Fix 2) ────────────────────────────
        sp_in = 2 + sp_feat_channels   # rgb_max + rgb_avg + scatter
        self.spatial_net_sharp  = _make_spatial_net(sp_in)
        self.log_temp_sharp     = nn.Parameter(torch.tensor(init_temp_sharp).log())

        # ── BRANCH COARSE: global focus (Fix 2) ──────────────────────────
        self.spatial_net_coarse = _make_spatial_net(sp_in)
        self.log_temp_coarse    = nn.Parameter(torch.tensor(init_temp_coarse).log())

        # ── BRANCH MIXER: per-sample blend weight (Fix 2) ─────────────────
        self.branch_mixer = nn.Sequential(
            nn.Linear(skel_grid_size, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1, bias=True),
            nn.Sigmoid(),
        )
        # Init to 0 -> sigmoid(0)=0.5 -> equal branch weight at start
        nn.init.zeros_(self.branch_mixer[-2].weight)
        nn.init.zeros_(self.branch_mixer[-2].bias)

        # ── SPATIAL-WISE GATE (magnitude only) ────────────────────────────
        self.spatial_gate = SpatialWiseGate(skel_channels)

        # ── CONFIDENCE GATE with jitter (Fix 3) ───────────────────────────
        conf_in = skel_grid_size + rgb_channels + num_parts  # 200+512+5 = 717
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

    @staticmethod
    def _gaussian_heatmaps(part_coords, H, W, sigma):
        B, P, _ = part_coords.shape
        dev = part_coords.device
        cx = (part_coords[:, :, 0] + 1) / 2 * (W - 1)  # (B, P)
        cy = (part_coords[:, :, 1] + 1) / 2 * (H - 1)
        yy = torch.arange(H, device=dev, dtype=torch.float32).view(1, 1, H, 1)
        xx = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, 1, W)
        return torch.exp(
            -((xx - cx.view(B,P,1,1))**2 + (yy - cy.view(B,P,1,1))**2)
            / (2 * sigma**2)
        )  # (B, P, H, W)

    def forward(self, rgb_feat, skel_grid, part_coords, norm_jitter, exp_type='normal'):
        B, C, H, W = rgb_feat.shape

        if exp_type == 'noise':
            skel_grid   = torch.randn_like(skel_grid)
            part_coords = torch.rand_like(part_coords) * 2 - 1
            # norm_jitter stays real — it is the corruption detector (Fix 3)
        elif exp_type == 'ones':
            skel_grid = torch.ones_like(skel_grid)
        elif exp_type == 'zeros':
            skel_grid = torch.zeros_like(skel_grid)

        skel_flat = skel_grid.view(B, -1)

        # ── CONFIDENCE GATE (Fix 3) ───────────────────────────────────────
        rgb_global = F.adaptive_avg_pool2d(rgb_feat, 1).view(B, -1)
        confidence = self.confidence_gate(
            torch.cat([skel_flat, rgb_global, norm_jitter], dim=1)
        )

        # ── CHANNEL ATTENTION ─────────────────────────────────────────────
        ch_attn = self.channel_attn(skel_flat).unsqueeze(-1).unsqueeze(-1)
        feat_ca = rgb_feat * ch_attn

        if exp_type == 'no_spatial':
            alpha = confidence.unsqueeze(-1).unsqueeze(-1)
            return rgb_feat + alpha * feat_ca, confidence

        # ── SPATIAL-WISE GATE (magnitude) ─────────────────────────────────
        gate_map = self.spatial_gate(skel_grid)

        # ── GAUSSIAN SCATTER (Fix 1) ──────────────────────────────────────
        skel_parts      = skel_grid.mean(dim=3)               # (B, K, 5)
        skel_parts_proj = self.part_feat_proj(skel_parts)     # (B, C_sp, 5)
        adj = (part_coords + self.coord_adjust(part_coords)).clamp(-1, 1)

        scatter_s = torch.einsum(
            'bcp,bphw->bchw', skel_parts_proj,
            self._gaussian_heatmaps(adj, H, W, self.log_sigma_sharp.exp())
        )
        scatter_c = torch.einsum(
            'bcp,bphw->bchw', skel_parts_proj,
            self._gaussian_heatmaps(adj, H, W, self.log_sigma_coarse.exp())
        )

        # ── DUAL-BRANCH SPATIAL ATTENTION (Fix 2) ─────────────────────────
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)

        T_s = self.log_temp_sharp.exp()
        T_c = self.log_temp_coarse.exp()

        sp_sharp  = torch.sigmoid(
            self.spatial_net_sharp(torch.cat([rgb_max, rgb_avg, scatter_s], dim=1)) / T_s
        )
        sp_coarse = torch.sigmoid(
            self.spatial_net_coarse(torch.cat([rgb_max, rgb_avg, scatter_c], dim=1)) / T_c
        )

        w       = self.branch_mixer(skel_flat).view(B, 1, 1, 1)
        sp_attn = w * sp_sharp + (1 - w) * sp_coarse

        # ── FUSION ────────────────────────────────────────────────────────
        skel_delta  = feat_ca * sp_attn * gate_map
        alpha       = confidence.unsqueeze(-1).unsqueeze(-1)
        output_feat = rgb_feat + alpha * skel_delta
        return output_feat, confidence


class Model(nn.Module):
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5,
                 exp_type='normal', proj_channels=8, sp_feat_channels=4,
                 init_sigma_sharp=3.0, init_sigma_coarse=8.0,
                 init_temp_sharp=0.3, init_temp_coarse=1.5,
                 consistency_weight=0.1, num_point=20, num_person=1):
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

        resnet       = models.resnet50(pretrained=pretrained)
        self.stem    = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc      = nn.Linear(resnet.fc.in_features, num_class)

        self.rgb_only_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(512, num_class),
        )

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

        self.cross_attn = CrossModalAttentionV17(
            rgb_channels=512, skel_channels=K,
            skel_grid_size=K * 5 * temporal_rgb_frames,
            reduction=4, num_parts=5,
            sp_feat_channels=sp_feat_channels,
            init_sigma_sharp=init_sigma_sharp,
            init_sigma_coarse=init_sigma_coarse,
            init_temp_sharp=init_temp_sharp,
            init_temp_coarse=init_temp_coarse,
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

    def _extract_part_coords(self, x_s):
        B, _, T, V, M = x_s.shape
        c = x_s[:,:2,:,:,:].mean(dim=4) if M > 1 else x_s[:,:2,:,:,0]  # (B, 2, T, V)

        # Per-sample normalize to [-1, 1] so this works for both:
        #   NW-UCLA (feeder pre-normalizes) and NTU-60 (raw Kinect meters)
        c_flat = c.reshape(B, 2, -1)
        c_min  = c_flat.min(dim=2, keepdim=True)[0].unsqueeze(2)   # (B,2,1,1)
        c_max  = c_flat.max(dim=2, keepdim=True)[0].unsqueeze(2)
        c = 2.0 * (c - c_min) / (c_max - c_min + 1e-6) - 1.0     # -> [-1, 1]

        frames, result = [max(0,T//4), T//2, min(T-1,3*T//4)], []
        for t in frames:
            ct = c[:,:,t,:].clone()
            ct[:,1,:] = -ct[:,1,:]  # Y-flip (Kinect Y-up -> image Y-down)
            result.append(torch.stack(
                [ct[:,:,g].mean(dim=2) for g in self.part_groups], dim=1
            ))
        return torch.stack(result, dim=0).mean(dim=0)  # (B, 5, 2)

    def _compute_skeleton_uncertainty(self, x_s):
        """Normalized jitter (B, 5) — proxy for skeleton detection quality.
        Scale-invariant: works for both NW-UCLA (normalized) and NTU-60 (meters).
        """
        B, _, T, V, M = x_s.shape
        c = x_s[:,:2,:,:,:].mean(dim=4) if M > 1 else x_s[:,:2,:,:,0]  # (B,2,T,V)

        # Normalize per-sample so jitter is scale-invariant across datasets
        c_flat = c.reshape(B, 2, -1)
        c_min  = c_flat.min(dim=2, keepdim=True)[0].unsqueeze(2)
        c_max  = c_flat.max(dim=2, keepdim=True)[0].unsqueeze(2)
        c = 2.0 * (c - c_min) / (c_max - c_min + 1e-6) - 1.0

        vel        = c[:,:,1:,:] - c[:,:,:-1,:]
        speed      = vel.pow(2).sum(dim=1).sqrt()           # (B, T-1, V)
        accel      = vel[:,:,1:,:] - vel[:,:,:-1,:]
        jitter     = accel.pow(2).sum(dim=1).mean(dim=1)   # (B, V)
        mean_sp_sq = speed.mean(dim=1).pow(2)               # (B, V)
        norm_j     = jitter / (mean_sp_sq + 1e-4)
        return torch.cat(
            [norm_j[:,g].mean(dim=1, keepdim=True) for g in self.part_groups],
            dim=1,
        )  # (B, 5)

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
            part_coords = self._extract_part_coords(x_s)
            norm_jitter = self._compute_skeleton_uncertainty(x_s)  # now used!
            _, feature_s = self.ctrgcn.extract_feature(x_s)

        skel_grid = self._build_skel_grid(feature_s.detach())

        x = self.stem(x_rgb)
        x = self.layer1(x)
        x = self.layer2(x)

        if self.training:
            rgb_only_logits = self.rgb_only_head(x.detach())

        x_fused, confidence = self.cross_attn(
            x, skel_grid, part_coords, norm_jitter, exp_type=self.exp_type
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
