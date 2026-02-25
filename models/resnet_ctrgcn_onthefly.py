import torch
import torch.nn as nn


# Import only the ResNet structure
from models.resnet_only import Model as ResNet

class Model(nn.Module):
    def __init__(self, num_class, pretrained=True):
        super(Model, self).__init__()
        
        # The processor will inject the frozen CTR-GCN instance here.
        # This keeps the model definition clean and avoids hardcoding paths.
        self.ctrgcn = ''
        
        self.resnet = ResNet(num_class=num_class, pretrained=pretrained)
        self.temporal_rgb_frames = 5
        
    def forward(self, x_s, x_rgb):
        # x_s: Skeleton data -> (B, C, T, V, M) = (B, 3, 52, 20, 1) usually
        # x_rgb: Base STROI image -> (B, 3, 224, 224)
        
        # 1. Extract feature activations from frozen CTR-GCN (mirrors gen_ucla_stroi_weighted.py)
        # CTR-GCN has two temporal stride-2 layers, so T: 52 -> 13. V stays at 20.
        with torch.no_grad():
            _, feature_s = self.ctrgcn.extract_feature(x_s)
            # feature_s: (B, C_new, T_new, V, M) e.g. (B, 256, 13, 20, 1)
            
            # L2 norm across channel dim — same as offline: (feat*feat).sum(dim=1)**0.5
            intensity = (feature_s * feature_s).sum(dim=1) ** 0.5  # (B, T_new, V, M)
            intensity = torch.abs(intensity)
            
            # Global min-max normalize to [0, 255] per sample (matching offline script)
            B = intensity.shape[0]
            flat = intensity.view(B, -1)
            f_min = flat.min(dim=1, keepdim=True)[0]
            f_max = flat.max(dim=1, keepdim=True)[0]
            diff = f_max - f_min
            diff[diff == 0] = 1e-6
            flat_norm = 255.0 * (flat - f_min) / diff
            intensity_norm = flat_norm.view_as(intensity)  # (B, T_new, V, M)
        
        # 2. Compute per-body-part scalar weight (same as offline: top-k mean / 127.0)
        # parts_v matches gen_ucla_stroi_weighted.py: [head, right_arm, left_arm, right_leg, left_leg]
        # But layout matches STROI rows: [hands, hands, legs, legs, head]
        parts_v = {
            'head': 3,      # head representative joint
            'hands': [11, 7],  # right hand, left hand representative joints
            'legs': [18, 14],  # right leg, left leg representative joints
        }
        
        def get_part_scalar(joint_indices):
            """Compute weight scalar per body part, matching offline script logic."""
            if not isinstance(joint_indices, list):
                joint_indices = [joint_indices]
            # Average across the representative joints for this part
            part_vals = []
            for v in joint_indices:
                part_feat = intensity_norm[:, :, v, 0]  # (B, T_new)
                # Top-k temporal mean (k = min(15, T_new)), matching offline
                k = min(15, part_feat.shape[1])
                topk_vals = torch.topk(part_feat, k, dim=1)[0]  # (B, k)
                part_vals.append(topk_vals.mean(dim=1))  # (B,)
            part_val = torch.stack(part_vals, dim=0).mean(dim=0)  # (B,)
            return part_val / 127.0  # scalar weight centered ~1.0
        
        w_head = get_part_scalar(parts_v['head'])    # (B,)
        w_hands = get_part_scalar(parts_v['hands'])  # (B,)
        w_legs = get_part_scalar(parts_v['legs'])     # (B,)
        
        # 3. Create per-body-part weight map (one scalar per row, matching offline)
        # STROI layout: Row 0-1: Arms(hands), Row 2-3: Legs, Row 4: Head
        weight_map = torch.ones(B, 1, 5, 1, device=x_rgb.device)
        weight_map[:, 0, 0, 0] = w_hands
        weight_map[:, 0, 1, 0] = w_hands
        weight_map[:, 0, 2, 0] = w_legs
        weight_map[:, 0, 3, 0] = w_legs
        weight_map[:, 0, 4, 0] = w_head
        
        # 4. Broadcast to full 224x224 (each row covers H/5 = ~45 pixels)
        weight_map_resized = torch.nn.functional.interpolate(weight_map, size=(224, 224), mode='nearest')
        
        # 5. Apply weight and run through ResNet
        x_rgb_weighted = x_rgb * weight_map_resized
        output = self.resnet(x_rgb_weighted)
        
        return output
