import torch
import torch.nn as nn


# Import only the ResNet structure
from models.resnet_only import Model as ResNet


class SkeletonAttention(nn.Module):
    """Lightweight learnable attention module.
    Takes raw skeleton feature scores per body part and learns 
    optimal attention weights via a 2-layer MLP + Sigmoid.
    """
    def __init__(self, num_parts=5, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_parts, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_parts),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, num_parts) raw feature scores
        # output: (B, num_parts) attention weights in [0, 1]
        return self.mlp(x)


class Model(nn.Module):
    def __init__(self, num_class, pretrained=True):
        super(Model, self).__init__()
        
        # The processor will inject the frozen CTR-GCN instance here.
        self.ctrgcn = ''
        
        self.resnet = ResNet(num_class=num_class, pretrained=pretrained)
        self.temporal_rgb_frames = 5
        
        # Learnable attention: converts 5 raw body-part scores → 5 attention weights
        self.attention = SkeletonAttention(num_parts=5, hidden_dim=16)
        
    def _extract_part_scores(self, intensity_norm):
        """Extract raw feature score per body part from normalized intensity.
        
        Uses the same joint mapping and top-k logic as the offline script,
        but returns raw scores instead of dividing by 127.
        
        Args:
            intensity_norm: (B, T_new, V, M) normalized intensity [0, 255]
        Returns:
            raw_scores: (B, 5) one score per STROI row order:
                        [hands_row0, hands_row1, legs_row2, legs_row3, head_row4]
        """
        # STROI layout matches offline gen_ucla_stroi_weighted.py:
        # parts_v = [3, 11, 7, 18, 14] → [head, right_hand, left_hand, right_leg, left_leg]
        # STROI rows: [Row0: right_arm, Row1: left_arm, Row2: right_leg, Row3: left_leg, Row4: head]
        joint_per_row = [11, 7, 18, 14, 3]  # map each STROI row to its representative joint
        
        B = intensity_norm.shape[0]
        scores = []
        for v in joint_per_row:
            part_feat = intensity_norm[:, :, v, 0]  # (B, T_new)
            k = min(15, part_feat.shape[1])
            topk_vals = torch.topk(part_feat, k, dim=1)[0]  # (B, k)
            scores.append(topk_vals.mean(dim=1))  # (B,)
        
        raw_scores = torch.stack(scores, dim=1)  # (B, 5)
        # Normalize to ~[0, 2] range (divide by 127) as baseline input to attention
        raw_scores = raw_scores / 127.0
        return raw_scores
        
    def forward(self, x_s, x_rgb):
        # x_s: Skeleton data -> (B, C, T, V, M) = (B, 3, 52, 20, 1) usually
        # x_rgb: Base STROI image -> (B, 3, 224, 224)
        
        # 1. Extract feature activations from frozen CTR-GCN
        with torch.no_grad():
            _, feature_s = self.ctrgcn.extract_feature(x_s)
            # feature_s: (B, C_new, T_new, V, M) e.g. (B, 256, 13, 20, 1)
            
            # L2 norm across channel dim
            intensity = (feature_s * feature_s).sum(dim=1) ** 0.5  # (B, T_new, V, M)
            intensity = torch.abs(intensity)
            
            # Global min-max normalize to [0, 255] per sample
            B = intensity.shape[0]
            flat = intensity.view(B, -1)
            f_min = flat.min(dim=1, keepdim=True)[0]
            f_max = flat.max(dim=1, keepdim=True)[0]
            diff = f_max - f_min
            diff[diff == 0] = 1e-6
            flat_norm = 255.0 * (flat - f_min) / diff
            intensity_norm = flat_norm.view_as(intensity)  # (B, T_new, V, M)
        
        # 2. Compute learnable attention from skeleton features
        raw_scores = self._extract_part_scores(intensity_norm.detach())  # (B, 5)
        attn_weights = self.attention(raw_scores)  # (B, 5) in [0, 1] via Sigmoid
        
        # Scale to [0.5, 1.5] range so attention modulates rather than suppresses
        attn_weights = 0.5 + attn_weights  # (B, 5) in [0.5, 1.5]
        
        # 3. Create weight map: (B, 1, 5, 1) → interpolate to (B, 1, 224, 224)
        weight_map = attn_weights.unsqueeze(1).unsqueeze(3)  # (B, 1, 5, 1)
        weight_map_resized = torch.nn.functional.interpolate(
            weight_map, size=(224, 224), mode='nearest'
        )
        
        # 4. Apply attention-weighted STROI and run through ResNet
        x_rgb_weighted = x_rgb * weight_map_resized
        output = self.resnet(x_rgb_weighted)
        
        return output
