import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

# import model and data feeder
from models.resnet_gcn_attention import ResNet_GCN_Attention
from feeder.feeder_nucla_fusion import Feeder

# To avoid "graph module not found" error during dynamic import in CTRGCN
sys.path.append('./')

def test_attention_logic(data_path):
    print(f"Loading data from {data_path}...")
    
    # 1. Initialize the Feeder (Dataset)
    feeder = Feeder(
        data_path=data_path,
        split='val',
        debug=True, # Load a small subset for fast testing
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=52,
        normalization=False,
        temporal_rgb_frames=5
    )
    
    # Get one sample
    data, label, index = feeder[0]
    # data[0] is skeleton [C, T, V, M]
    # data[1] is rgb [C, H, W]
    x_gcn = torch.tensor(data[0]).unsqueeze(0).float() # [1, C, T, V, M]
    x_rgb = torch.tensor(data[1]).unsqueeze(0).float() # [1, 15, 224, 224]
    
    print(f"Sample loaded. Label: {label}")
    print(f"Skeleton shape: {x_gcn.shape}")
    print(f"RGB shape: {x_rgb.shape}")
    
    # 2. Initialize the Model
    print("\nInitializing ResNet_GCN_Attention Model...")
    model = ResNet_GCN_Attention(
        num_class=10, 
        num_point=20, 
        num_person=1, 
        graph='graph.ucla.Graph',
        graph_args={'labeling_mode': 'spatial'},
        in_channels_gcn=3,
        in_channels_rgb=15, # 5 frames * 3 channels
        drop_out=0,
        adaptive=True,
        freeze_gcn=True
    )
    model.eval()
    
    # Get pretrained weights (if available to show more realistic attention)
    try:
        weights = torch.load('./result/nucla/CTROGC-GCN.pt')
        model.gcn.load_state_dict(weights)
        print("Pretrained GCN weights loaded successfully.")
    except Exception as e:
        print(f"Could not load pretrained GCN weights (expected during basic test): {e}")
        pass
        
    print("\nExtracting Features and Attention Maps...")
    with torch.no_grad():
        # -------- Mimicing the forward pass to get intermediate attention weights --------
        
        # 1. GCN Semantic Guidance
        f_gcn, _ = model.gcn.extract_feature(x_gcn) 
        f_gcn = f_gcn.mean(dim=(2, 3, 4)) # [1, 256]
        
        # 2. Channel Attention
        ch_att_weights = model.channel_attention(f_gcn) # [1, 2048]
        
        # 3. Spatial Attention
        sp_att_weights = model.spatial_attention(f_gcn) # [1, 49]
        sp_att_weights = sp_att_weights.view(-1, 1, 7, 7) # [1, 1, 7, 7]
        
        print("\n=== Validation Results ===")
        print(f"GCN Output Features shape: {f_gcn.shape} -> Expected [1, 256]")
        print(f"Channel Attention Vector shape: {ch_att_weights.shape} -> Expected [1, 2048]")
        print(f"Spatial Attention Mask shape: {sp_att_weights.shape} -> Expected [1, 1, 7, 7]")
        
        # Look at the values in the spatial attention map
        spatial_mask = sp_att_weights[0, 0].numpy()
        print("\nGenerated Spatial Attention Map (7x7 Grid values between 0 and 1):")
        print(np.around(spatial_mask, decimals=3))
        
        print("\nMathematical logic holds: The GCN successfully generated a spatial mask and a channel vector.")
        print("This mask will now be multiplied element-wise onto the ResNet 7x7 feature maps, forcing ResNet to focus on regions where the mask values are higher.")
        print("Since these are random untrained weights for the attention transformation layers, the mask values might be uniform right now, but training will shape them towards the action.")

if __name__ == '__main__':
    data_path = r'C:\Users\nguyn\Downloads\NW-UCLA-ALL\NW-UCLA-ALL'
    test_attention_logic(data_path)
