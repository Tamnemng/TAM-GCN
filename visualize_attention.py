import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchlight.torchlight.io import import_class
import cv2

# Hardcoded imports for standard TAM setup
from models.resnet_gcn_attention import ResNet_GCN_Attention

def get_parser():
    parser = argparse.ArgumentParser(description='Visualize Cross-Modal Attention Distribution')
    parser.add_argument('--config', '-c', default='config/nucla/cross_modal.yaml', help='path to the configuration file')
    parser.add_argument('--weights', default='result/nucla/cross_model.pt', help='path to the model weights')
    parser.add_argument('--data_path', default='../../input/datasets/nguyenductamhehe/nwucla/NW-UCLA-ALL', help='path to the local NW-UCLA dataset')
    parser.add_argument('--output_dir', default='./visualizations', help='where to save output heatmaps')
    parser.add_argument('--sample_idx', type=int, default=0, help='Which sample in the val set to visualize')
    return parser.parse_args()

def load_data(config, data_path):
    Feeder = import_class(config['feeder'])
    test_args = config['test_feeder_args']
    test_args['data_path'] = data_path
    
    dataset = Feeder(**test_args)
    return dataset

def draw_heatmap(rgb_frames, attention_weights, output_path):
    # rgb_frames (C, T, H, W) where C=3, T=5 -> We want to shape it visually
    # attention_weights will just be a set of scales applied to the resnet layers (usually channel-wise).
    # Since resnet applies spatial convs later, the true spatial heatmap requires Grad-CAM.
    # For now, let's visualize the raw channel activation profile by GCN vs without GCN.
    
    # We will use Grad-CAM conceptually for the last conv block of ResNet to see where it looks
    pass

def main():
    args = get_parser()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading Model...")
    model_args = config['model_args']
    model = import_class(config['model'])(**model_args)
    
    # Load Weights
    print(f"Loading weights from {args.weights}")
    weights = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(weights)
    model.eval()
    
    feature_maps = []
    attention_scales = []
    
    def hook_feature(module, input, output):
        feature_maps.append(output.detach())
        
    def hook_attention(module, input, output):
        attention_scales.append(output.detach())
        
    model.resnet.layer4.register_forward_hook(hook_feature)
    model.attention_transform.register_forward_hook(hook_attention)
    
    print("Loading DataLoader...")
    # Load the TRAIN split to find 'v02' samples
    Feeder = import_class(config['feeder'])
    train_args = config['train_feeder_args']
    train_args['data_path'] = args.data_path
    train_args['split'] = 'train'
    dataset = Feeder(**train_args)
    
    # Find specific user sample
    target_sample = 'a12_s10_e02_v02'
    if target_sample in dataset.sample_name:
        sample_idx = dataset.sample_name.index(target_sample)
    else:
        print(f"Sample {target_sample} not found! Fallback to index 10")
        sample_idx = 10
        
    data, label, idx = dataset[sample_idx]
    ske_data = torch.tensor(data[0]).unsqueeze(0).float()
    rgb_data = torch.tensor(data[1]).unsqueeze(0).float()
    
    print(f"Running Inference on: {dataset.sample_name[sample_idx]}...")
    with torch.no_grad():
        out = model(ske_data, rgb_data)
        
    print(f"Predicted Class: {out.argmax(-1).item()}, True Class: {label}")
    
    # Feature Map is shape (1, 2048, H, W)
    fmap = feature_maps[0]
    # Attention array is (1, 2048) -> applied to channels
    att = attention_scales[0]
    
    # Generate weighted CAM (Class Activation Map-like)
    # Multiply the spatial 2D feature maps by the GCN channel attention coefficients
    weighted_fmap = fmap * att.view(1, -1, 1, 1)
    
    plt.figure(figsize=(10, 4))
    plt.plot(att.squeeze().numpy()[:200]) # Plot first 200 channels
    plt.title("GCN Channel Attention Multipliers (First 200 Channels of ResNet50)")
    plt.xlabel("Channel Index")
    plt.ylabel("Multiplier Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'gcn_channel_attention.png'))
    plt.close()
    
    # We collapse the (1, 2048, H', W') to (1, H', W') array representing the final spatial heatmap
    spatial_map = weighted_fmap.mean(dim=1).squeeze().numpy()
    spatial_map = np.maximum(spatial_map, 0)
    spatial_map = spatial_map / np.max(spatial_map)
    spatial_map_resized = cv2.resize(spatial_map, (rgb_data.shape[-1], rgb_data.shape[-2]))
    heatmap = cv2.applyColorMap(np.uint8(255 * spatial_map_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Plot 5 Frames overlay
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    rgb_numpy = rgb_data.squeeze().numpy() # (15, 224, 224)
    # The normalisation usually is ImageNet normalisation, but we can approximate it back for display
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    
    for t in range(5):
        # RGB early fusion packs 15 channels (5 frames * 3 colors)
        frame = rgb_numpy[t*3:(t+1)*3, :, :] # (3, 224, 224)
        frame = frame * std + mean
        frame = np.clip(frame, 0, 1)
        frame = frame.transpose(1, 2, 0) # (224, 224, 3)
        
        # Combine true image and spatial heatmap
        # NOTE: The heatmap is identical for all 5 frames because Early Fusion (15-chan input)
        # collapses the temporal dimension at the very first layer of ResNet!
        superimposed_img = heatmap * 0.4 + frame * 0.6
        superimposed_img = superimposed_img / np.max(superimposed_img)
        
        axes[t].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB) if False else superimposed_img)
        axes[t].axis('off')
        axes[t].set_title(f"Time T={t} (Aggregated Spatial Attn)")
        
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'spatial_heatmap_frames.png'))
    print(f"Saved visualizations to {args.output_dir}")

if __name__ == '__main__':
    main()
