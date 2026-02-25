import torch
import torch.nn as nn
import numpy as np

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
        
    def map_15_to_5(self, w_15):
        # Maps 15 STGCN frames to 5 STROI frames by averaging every 3 frames
        # w_15 shape: (B, 15)
        return w_15.view(w_15.shape[0], 5, 3).mean(dim=2)

    def forward(self, x_s, x_rgb):
        # x_s: Skeleton data -> (B, C, T, V, M) = (B, 3, 52, 20, 1) usually
        # x_rgb: Base STROI image -> (B, 3, 224, 224)
        
        # 1. Extract feature activations directly from the frozen CTR-GCN.
        with torch.no_grad():
            _, feature_s = self.ctrgcn.extract_feature(x_s)
            feature_s = torch.abs(feature_s)  # Make feature activations positive
            
        # 2. Select exactly 15 key frames from the 52 available (MMNet original logic)
        indices = np.linspace(0, 51, 15, dtype=int)
        feature_s = feature_s[:, :, indices, :, :]
        
        # 3. Target the specific body parts (NW-UCLA 20 joints)
        head = [2]
        hands = [4, 5, 6, 7, 8, 9, 10, 11]
        legs = [12, 13, 14, 15, 16, 17, 18, 19]
        
        B = feature_s.shape[0]
        
        def get_part_weight(part_indices):
            # feature_s: (B, C, 15, V, M)
            part_feat = feature_s[:, :, :, part_indices, :]
            # Mean pooling over Channel(1), Selected Joints(3), Person(4)
            part_feat = torch.mean(part_feat, dim=(1, 3, 4)) # -> (B, 15)
            
            # Local Min-Max normalization individually per batch element
            min_val = part_feat.min(dim=1, keepdim=True)[0]
            max_val = part_feat.max(dim=1, keepdim=True)[0]
            diff = max_val - min_val
            diff[diff == 0] = 1e-6
            return (part_feat - min_val) / diff # -> Soft attention weight (B, 15) in standard bounds [0, 1]
            
        w_h_15 = get_part_weight(head)
        w_ha_15 = get_part_weight(hands)
        w_l_15 = get_part_weight(legs)
        
        # 4. Map 15 temporal weights down to 5 spatial RGB columns
        w_h_5 = self.map_15_to_5(w_h_15)   # (B, 5)
        w_ha_5 = self.map_15_to_5(w_ha_15) # (B, 5)
        w_l_5 = self.map_15_to_5(w_l_15)   # (B, 5)
        
        # 5. Create 5x5 dynamic weight map
        # STROI visual layout: Width/Columns = frame time. Height/Rows = body part structure
        # Row 0: Left Arm (hands)
        # Row 1: Right Arm (hands)
        # Row 2: Left Leg (legs)
        # Row 3: Right Leg (legs)
        # Row 4: Head (head)
        weight_map = torch.zeros(B, 1, 5, 5, device=x_rgb.device)
        
        weight_map[:, 0, 0, :] = w_ha_5
        weight_map[:, 0, 1, :] = w_ha_5
        weight_map[:, 0, 2, :] = w_l_5
        weight_map[:, 0, 3, :] = w_l_5
        weight_map[:, 0, 4, :] = w_h_5
        
        # 6. Interpolate the absolute 5x5 matrix into the full 224x224 RGB canvas smoothly mimicking pixels
        # Nearest guarantees no bleeding over the rigidly pasted boxes from gen_ucla_stroi.py
        weight_map_resized = torch.nn.functional.interpolate(weight_map, size=(224, 224), mode='nearest')
        
        # 7. Dynamically scale the single RGB view, and run End-to-End training through ResNet. 
        x_rgb_weighted = x_rgb * weight_map_resized
        output = self.resnet(x_rgb_weighted)
        
        return output
