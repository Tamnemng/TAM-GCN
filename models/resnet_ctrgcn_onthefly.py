import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CrossModalAttention(nn.Module):
    """Cross-Modal Attention: injects skeleton features into ResNet feature maps.
    
    Unlike MMnet which multiplies attention at pixel level (before any feature learning),
    this module fuses skeleton and RGB at the FEATURE level — after ResNet has already
    extracted meaningful visual features.
    
    Performs both:
    - Channel attention: skeleton features select WHICH feature channels are important
    - Spatial attention: skeleton spatiotemporal grid highlights WHERE to look
    
    Uses residual connection to preserve original RGB features.
    """
    def __init__(self, rgb_channels, skel_grid_size=25, reduction=4):
        """
        Args:
            rgb_channels: number of channels in the ResNet feature map (e.g. 512 for layer2)
            skel_grid_size: flattened size of skeleton grid (5 parts × 5 times = 25)
            reduction: channel reduction ratio for the channel attention MLP
        """
        super().__init__()
        
        # Channel attention: skeleton → which channels matter
        self.channel_attn = nn.Sequential(
            nn.Linear(skel_grid_size, rgb_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // reduction, rgb_channels),
            nn.Sigmoid()
        )
        
        # Spatial attention: skeleton grid → where to look in feature map
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, rgb_feat, skel_grid):
        """
        Args:
            rgb_feat: (B, C, H, W) ResNet feature map
            skel_grid: (B, 1, 5, 5) spatiotemporal skeleton feature grid
        Returns:
            fused_feat: (B, C, H, W) feature map with skeleton info injected
        """
        B, C, H, W = rgb_feat.shape
        
        # 1. Channel attention from skeleton
        skel_flat = skel_grid.view(B, -1)                      # (B, 25)
        ch_attn = self.channel_attn(skel_flat)                  # (B, C)
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)           # (B, C, 1, 1)
        
        # 2. Spatial attention from skeleton grid
        sp_attn = F.interpolate(skel_grid, size=(H, W), mode='bilinear', align_corners=False)  # (B, 1, H, W)
        sp_attn = self.spatial_proj(sp_attn)                    # (B, 1, H, W)
        
        # 3. Combined cross-modal modulation + residual
        modulated = rgb_feat * ch_attn * sp_attn                # (B, C, H, W)
        return rgb_feat + modulated                             # residual: keep original + add skeleton-guided features


class Model(nn.Module):
    def __init__(self, num_class, pretrained=True, temporal_rgb_frames=5):
        super(Model, self).__init__()
        
        # The processor will inject the frozen CTR-GCN instance here.
        self.ctrgcn = ''
        self.temporal_rgb_frames = temporal_rgb_frames
        
        # ResNet-50 backbone — we'll run it stage by stage
        resnet = models.resnet50(pretrained=pretrained)
        
        # Split ResNet into stages for mid-level injection
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )                                                       # → (B, 64, 56, 56)
        self.layer1 = resnet.layer1                              # → (B, 256, 56, 56)
        self.layer2 = resnet.layer2                              # → (B, 512, 28, 28)
        # ↑ Cross-modal injection point ↑
        self.layer3 = resnet.layer3                              # → (B, 1024, 14, 14)
        self.layer4 = resnet.layer4                              # → (B, 2048, 7, 7)
        self.avgpool = resnet.avgpool                            # → (B, 2048, 1, 1)
        self.fc = nn.Linear(resnet.fc.in_features, num_class)    # → (B, num_class)
        
        # Cross-modal attention at layer2 output (512 channels)
        self.cross_attn = CrossModalAttention(
            rgb_channels=512,
            skel_grid_size=5 * temporal_rgb_frames,  # 5 parts × 5 frames = 25
            reduction=4
        )
        
        # Body part to joint index mapping (matches STROI row order)
        self.part_joints = [3, 7, 11, 14, 18]
        
    def _build_skel_grid(self, intensity_norm):
        """Build (B, 1, 5, T_frames) spatiotemporal skeleton feature grid.
        
        Rows = 5 body parts, Cols = T_frames temporal bins.
        Each cell = avg skeleton feature intensity for that (part, time).
        """
        B, T_new, V, M = intensity_norm.shape
        T_frames = self.temporal_rgb_frames
        
        # Extract per-joint intensities → pool into grid
        part_features = []
        for v in self.part_joints:
            part_features.append(intensity_norm[:, :, v, 0])  # (B, T_new)
        part_features = torch.stack(part_features, dim=1)     # (B, 5, T_new)
        
        # Adaptive pool T_new → T_frames temporal bins
        part_features = F.adaptive_avg_pool1d(part_features, T_frames)  # (B, 5, T_frames)
        
        # Normalize
        part_features = part_features / 127.0
        
        return part_features.unsqueeze(1)  # (B, 1, 5, T_frames)
        
    def forward(self, x_s, x_rgb):
        # x_s: (B, C, T, V, M) = (B, 3, 52, 20, 1)
        # x_rgb: (B, 3, 224, 224) base STROI image
        
        # ===== 1. Skeleton branch: extract spatiotemporal features =====
        with torch.no_grad():
            _, feature_s = self.ctrgcn.extract_feature(x_s)
            
            # L2 norm across channel dim
            intensity = (feature_s * feature_s).sum(dim=1) ** 0.5
            intensity = torch.abs(intensity)
            
            # Per-sample min-max normalize to [0, 255]
            B = intensity.shape[0]
            flat = intensity.view(B, -1)
            f_min = flat.min(dim=1, keepdim=True)[0]
            f_max = flat.max(dim=1, keepdim=True)[0]
            diff = f_max - f_min
            diff[diff == 0] = 1e-6
            flat_norm = 255.0 * (flat - f_min) / diff
            intensity_norm = flat_norm.view_as(intensity)
        
        # Build skeleton spatiotemporal grid
        skel_grid = self._build_skel_grid(intensity_norm.detach())  # (B, 1, 5, 5)
        
        # ===== 2. RGB branch: ResNet stages with cross-modal injection =====
        x = self.stem(x_rgb)       # (B, 64, 56, 56)
        x = self.layer1(x)         # (B, 256, 56, 56)
        x = self.layer2(x)         # (B, 512, 28, 28)
        
        # ★ Cross-modal injection: skeleton features meet RGB features 
        x = self.cross_attn(x, skel_grid)  # (B, 512, 28, 28) — fused
        
        x = self.layer3(x)         # (B, 1024, 14, 14)
        x = self.layer4(x)         # (B, 2048, 7, 7)
        x = self.avgpool(x)        # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)    # (B, 2048)
        output = self.fc(x)        # (B, num_class)
        
        return output
