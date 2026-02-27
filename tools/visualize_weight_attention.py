"""
Visualization: Skeleton-Weighted Attention Heatmaps on STROI Images

Generates 3 visualizations per sample:
  1. STROI Image (Input)
  2. Skeleton Weight Vector (5x1 attention scores)
  3. Attention Overlay (Weights overlaid on STROI)
  4. Grad-CAM (ResNet focus after weighting)

Usage:
    python tools/visualize_weight_attention.py \
        --weights ./work_dir/nucla/resnet_weight_v2/best_model.pt \
        --ctrgcn_weights ./result/nucla/CTROGC-GCN.pt \
        --data_path C:/Users/nguyn/Downloads/NW-UCLA-ALL/NW-UCLA-ALL \
        --rgb_path C:/ucla_stroi/ \
        --num_samples 10 \
        --output_dir ./vis_weight_attn/
"""
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append(os.getcwd())

from models.ctrgcn import Model as CTR_GCN_Model
from models.resnet_ctrgcn_weight import Model as WeightModel
from feeder.feeder_nucla_fused_ctr_resnet import Feeder

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_CLASS = 10
LABEL_NAMES = {
    0: 'Pick up with one hand', 1: 'Pick up with two hands',
    2: 'Drop trash', 3: 'Walk around', 4: 'Sit down',
    5: 'Stand up', 6: 'Donning', 7: 'Doffing',
    8: 'Throw', 9: 'Carry',
}
ROW_NAMES = ['Head (Row0)', 'L_Hand (Row1)', 'R_Hand (Row2)', 'L_Leg (Row3)', 'R_Leg (Row4)']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--ctrgcn_weights', type=str, default='./result/nucla/CTROGC-GCN.pt')
    parser.add_argument('--data_path', type=str, default='C:/Users/nguyn/Downloads/NW-UCLA-ALL/NW-UCLA-ALL')
    parser.add_argument('--rgb_path', type=str, default='C:/ucla_stroi/')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./vis_weight_attn/')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                        help='Specific sample indices to visualize')
    return parser.parse_args()


def load_weights_robust(path, device):
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in raw:
                raw = raw[key]
                break
    new_state = OrderedDict()
    for k, v in raw.items():
        new_state[k.replace('module.', '')] = v
    return new_state


def load_model(weights_path, ctrgcn_weights_path, device):
    model = WeightModel(num_class=NUM_CLASS, pretrained=False).to(device)

    # Inject CTR-GCN
    ctrgcn = CTR_GCN_Model(
        num_class=NUM_CLASS, num_point=20, num_person=1,
        graph='graph.ucla.Graph', graph_args={'labeling_mode': 'spatial'},
    ).to(device)
    ctrgcn_state = load_weights_robust(ctrgcn_weights_path, device)
    ctrgcn.load_state_dict(ctrgcn_state, strict=False)
    ctrgcn.eval()
    for p in ctrgcn.parameters():
        p.requires_grad = False
    model.ctrgcn = ctrgcn

    # Load trained weights
    state_dict = load_weights_robust(weights_path, device)
    model_state = model.state_dict()
    loaded = 0
    for k, v in state_dict.items():
        if k in model_state and model_state[k].size() == v.size():
            model_state[k] = v
            loaded += 1
    model.load_state_dict(model_state, strict=False)
    model.eval()
    print(f"  ✓ Model loaded ({loaded} params): {weights_path}")
    return model


def denormalize_image(tensor):
    """Convert normalized tensor back to displayable image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def extract_attention_maps(model, skeleton, rgb, device):
    """Run forward pass and extract intermediate attention weights + Grad-CAM."""
    skeleton = torch.as_tensor(skeleton).unsqueeze(0).float().to(device)
    rgb = torch.as_tensor(rgb).unsqueeze(0).float().to(device)
    rgb.requires_grad = False

    # ===== Step 1: Compute weights from CTR-GCN features =====
    with torch.no_grad():
        _, feature_s = model.ctrgcn.extract_feature(skeleton)
        intensity = (feature_s * feature_s).sum(dim=1) ** 0.5
        intensity = torch.abs(intensity)
        B = intensity.shape[0]
        flat = intensity.view(B, -1)
        f_min = flat.min(dim=1, keepdim=True)[0]
        f_max = flat.max(dim=1, keepdim=True)[0]
        diff = f_max - f_min
        diff[diff == 0] = 1e-6
        flat_norm = 255.0 * (flat - f_min) / diff
        intensity_norm = flat_norm.view_as(intensity)

        raw_scores = model._extract_part_scores(intensity_norm)  # (1, 5)
        attn_weights = model.attention(raw_scores)               # (1, 5)
        mod_weights = 0.5 + attn_weights                         # (1, 5)
        
        weight_map = mod_weights.unsqueeze(1).unsqueeze(3)       # (1, 1, 5, 1)
        weight_map_resized = F.interpolate(weight_map, size=(224, 224), mode="nearest")

    # ===== Step 2: Grad-CAM on ResNet layer4 =====
    gradcam_features = {}

    def save_activation(name):
        def hook(module, input, output):
            gradcam_features[name] = output
        return hook

    def save_gradient(name):
        def hook(module, input, output):
            gradcam_features[name + '_grad'] = output[0]
        return hook

    handle_fwd = model.resnet.model.layer4.register_forward_hook(save_activation('layer4'))
    handle_bwd = model.resnet.model.layer4.register_full_backward_hook(save_gradient('layer4'))

    output = model(skeleton, rgb)
    pred_class = output.argmax(dim=1).item()

    # Backward for predicted class
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, pred_class] = 1.0
    output.backward(gradient=one_hot, retain_graph=False)

    handle_fwd.remove()
    handle_bwd.remove()

    # Compute Grad-CAM
    feats = gradcam_features['layer4'].detach()          # (1, 2048, 7, 7)
    grads = gradcam_features['layer4_grad'].detach()     # (1, 2048, 7, 7)
    weights = grads.mean(dim=[2, 3], keepdim=True)       # (1, 2048, 1, 1)
    cam = (weights * feats).sum(dim=1, keepdim=True)     # (1, 1, 7, 7)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    return {
        'raw_scores': raw_scores.squeeze().cpu().numpy(),
        'attn_weights': attn_weights.squeeze().cpu().numpy(),
        'mod_weights': mod_weights.squeeze().cpu().numpy(),
        'weight_map_resized': weight_map_resized.squeeze().cpu().numpy(),
        'gradcam': cam,
        'pred_class': pred_class,
        'output_logits': output.detach().cpu().numpy().squeeze(),
    }


def plot_sample(rgb_tensor, label, attn_data, save_path, sample_name=''):
    """Create a 4-panel visualization figure."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f'{sample_name}  |  GT: {LABEL_NAMES[label]}  |  Pred: {LABEL_NAMES[attn_data["pred_class"]]}',
        fontsize=14, fontweight='bold',
        color='green' if label == attn_data['pred_class'] else 'red'
    )

    # 1. Original STROI image
    img = denormalize_image(rgb_tensor)
    axes[0].imshow(img)
    axes[0].set_title('STROI Image (Input)')
    axes[0].axis('off')

    # 2. Skeleton weights
    mod_weights = attn_data['mod_weights'].reshape(5, 1)
    im1 = axes[1].imshow(mod_weights, cmap='inferno', vmin=0.5, vmax=1.5, aspect='auto')
    axes[1].set_title('Skeleton Multiplier (0.5 - 1.5)')
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(ROW_NAMES, fontsize=10)
    axes[1].set_xticks([])
    for i in range(5):
        axes[1].text(0, i, f"{mod_weights[i,0]:.3f}", ha="center", va="center", color="white", fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # 3. Weight mask map over STROI
    weight_map_resized = attn_data['weight_map_resized']
    axes[2].imshow(img)
    axes[2].imshow(weight_map_resized, cmap='inferno', alpha=0.4, vmin=0.5, vmax=1.5)
    axes[2].set_title('Weighted STROI Overlay')
    axes[2].axis('off')

    # 4. Grad-CAM (final layer — where ResNet focuses)
    axes[3].imshow(img)
    axes[3].imshow(attn_data['gradcam'], cmap='jet', alpha=0.5)
    axes[3].set_title('Grad-CAM (ResNet Focus)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def log_attention_weights(model):
    """Log the MLP weights once."""
    print('\n' + '=' * 60)
    print('  SKELETON ATTENTION WEIGHTS (MLP)')
    print('=' * 60)
    try:
        w1 = model.attention.mlp[0].weight.detach().cpu().numpy()
        b1 = model.attention.mlp[0].bias.detach().cpu().numpy()
        w2 = model.attention.mlp[2].weight.detach().cpu().numpy()
        b2 = model.attention.mlp[2].bias.detach().cpu().numpy()
        print(f'  Layer 1 (Linear V->H): weight_shape={w1.shape}, bias_shape={b1.shape}')
        print(f'  Layer 2 (Linear H->V): weight_shape={w2.shape}, bias_shape={b2.shape}')
        print(f'  => This maps 5 raw scores to 5 attention weights in [0, 1].')
    except Exception as e:
        print(f"Could not load MLP weights for logging: {e}")
    print('=' * 60)


def log_attention_details(sample_name, label, attn_data):
    """Print detailed numerical info for one sample."""
    raw = attn_data['raw_scores']
    attn = attn_data['attn_weights']
    mod = attn_data['mod_weights']
    logits = attn_data['output_logits']
    pred = attn_data['pred_class']
    
    print(f'\n{"~" * 60}')
    print(f'  SAMPLE: {sample_name}  |  GT: {LABEL_NAMES[label]}  |  Pred: {LABEL_NAMES[pred]}')
    print(f'{"~" * 60}')
    
    # 1. Weights
    print(f'\n  [1] SKELETON ATTENTION WEIGHTS:')
    print(f'    {"Part":15s} | {"Raw Score":>10s} | {"Attn(0~1)":>10s} | {"Final(0.5~1.5)":>14s}')
    print(f'    {"-"*15}-+-{"-"*10}-+-{"-"*10}-+-{"-"*14}')
    for i, name in enumerate(ROW_NAMES):
        print(f'    {name:15s} | {raw[i]:10.4f} | {attn[i]:10.4f} | {mod[i]:14.4f}')
    
    # 2. Logits
    print(f'\n  [2] OUTPUT LOGITS:')
    sorted_idx = logits.argsort()[::-1]
    for rank, i in enumerate(sorted_idx[:5]):
        marker = ' <<< GT' if i == label else ''
        marker += ' <<< PRED' if i == pred else ''
        print(f'    #{rank+1}: {LABEL_NAMES[i]:25s} = {logits[i]:+.3f}{marker}')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print('  Skeleton-Weighted Attention Visualization')
    print('=' * 60)

    # Load model
    model = load_model(args.weights, args.ctrgcn_weights, DEVICE)
    
    # Log weights ONCE
    log_attention_weights(model)

    # Load val data
    dataset = Feeder(
        data_path=args.data_path, label_path='val',
        rgb_path=args.rgb_path, temporal_rgb_frames=5,
        random_choose=False, random_shift=False, random_move=False,
        window_size=52, normalization=False,
    )
    print(f"  Val samples: {len(dataset)}")

    # Select sample indices
    if args.sample_indices:
        indices = args.sample_indices
    else:
        # Pick diverse samples: one from each class
        label_to_idx = {}
        for i in range(len(dataset)):
            _, _, label = dataset[i]
            if isinstance(label, (int, np.integer)):
                l = int(label)
            else:
                l = label
            if l not in label_to_idx:
                label_to_idx[l] = i
            if len(label_to_idx) >= min(args.num_samples, NUM_CLASS):
                break
        indices = list(label_to_idx.values())
        # Fill remaining with random
        if len(indices) < args.num_samples:
            remaining = [i for i in range(len(dataset)) if i not in indices]
            np.random.seed(42)
            extra = np.random.choice(remaining, min(args.num_samples - len(indices), len(remaining)), replace=False)
            indices.extend(extra.tolist())

    print(f"\n>>> Visualizing {len(indices)} samples...")

    for idx in indices:
        skeleton, rgb, label = dataset[idx]
        if isinstance(label, (int, np.integer)):
            label = int(label)

        # Get sample name
        info = dataset.data_dict[idx % len(dataset.data_dict)]
        sample_name = info.get('file_name', f'sample_{idx}')

        # Extract attention maps
        attn_data = extract_attention_maps(model, skeleton, rgb, DEVICE)
        
        # Log detailed numbers
        log_attention_details(sample_name, label, attn_data)

        # Plot
        save_path = os.path.join(args.output_dir, f'{sample_name}_attn.png')
        plot_sample(rgb, label, attn_data, save_path, sample_name)

    print(f"\n[DONE] Saved {len(indices)} visualizations to {args.output_dir}")


if __name__ == '__main__':
    main()
