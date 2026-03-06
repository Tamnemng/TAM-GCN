"""
Visualization: Cross-Modal Attention Heatmaps on STROI Images — V2

V2 model improvements over V0:
  1. Conv1d(256->1) learnable channel projection (replaces fixed L2 norm)
  2. Conv1d(20->5) learnable joint-to-part grouping (replaces fixed index pick)
     Initialized with anatomical groups: Torso[0-3], L-Arm[4-7], R-Arm[8-11],
     L-Leg[12-15], R-Leg[16-19]
  3. ConvTranspose2d learned upsampling 5x5->14x14->28x28
     (replaces bilinear F.interpolate)

Generates 5 visualizations per sample:
  1. STROI Image       — input RGB image
  2. Skeleton Grid     — raw 5×5 spatiotemporal feature grid (learned projection)
  3. Learned Upsample  — 28×28 skeleton map from ConvTranspose2d
  4. Cross-Attention   — spatial attention map from CrossModalAttention
  5. Grad-CAM          — where ResNet focuses AFTER cross-attention injection

Usage:
    python tools/visualize_cross_attentionv2.py \
        --weights ./work_dir/nucla/resnet_onthefly_v2/best_model.pt \
        --ctrgcn_weights ./result/nucla/CTROGC-GCN.pt \
        --data_path C:/Users/nguyn/Downloads/NW-UCLA-ALL/NW-UCLA-ALL \
        --rgb_path C:/ucla_stroi/ \
        --num_samples 10 \
        --output_dir ./vis_cross_attn_v2/
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
from models.resnet_ctrgcn_ontheflyv2 import Model as CrossModalModel
from feeder.feeder_nucla_fused_ctr_resnet import Feeder

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_CLASS = 10
LABEL_NAMES = {
    0: 'Pick up with one hand', 1: 'Pick up with two hands',
    2: 'Drop trash', 3: 'Walk around', 4: 'Sit down',
    5: 'Stand up', 6: 'Donning', 7: 'Doffing',
    8: 'Throw', 9: 'Carry',
}
PART_NAMES = ['Torso', 'L_Arm', 'R_Arm', 'L_Leg', 'R_Leg']

# V2: Conv1d(20->5) learns joint-to-part, initialized with these groups
PART_GROUPS = [
    [0, 1, 2, 3],      # Torso
    [4, 5, 6, 7],      # Left Arm
    [8, 9, 10, 11],    # Right Arm
    [12, 13, 14, 15],  # Left Leg
    [16, 17, 18, 19],  # Right Leg
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--ctrgcn_weights', type=str, default='./result/nucla/CTROGC-GCN.pt')
    parser.add_argument('--data_path', type=str, default='C:/Users/nguyn/Downloads/NW-UCLA-ALL/NW-UCLA-ALL')
    parser.add_argument('--rgb_path', type=str, default='C:/ucla_stroi/')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./vis_cross_attn_v2/')
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
    model = CrossModalModel(num_class=NUM_CLASS, pretrained=False).to(device)

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
    model.load_state_dict(model_state)
    model.eval()
    print(f"  ✓ V2 Model loaded ({loaded} params): {weights_path}")
    return model


def denormalize_image(tensor):
    """Convert normalized tensor back to displayable image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def extract_attention_maps(model, skeleton, rgb, device):
    """Run forward pass and extract intermediate attention maps + Grad-CAM.

    V2: Uses _build_skel_grid_v2(feature_s) with learnable Conv1d projection
        and Conv1d joint-to-part grouping.
        CrossModalAttention uses ConvTranspose2d learned upsampling.
    """
    skeleton = torch.as_tensor(skeleton).unsqueeze(0).float().to(device)
    rgb = torch.as_tensor(rgb).unsqueeze(0).float().to(device)
    rgb.requires_grad = False

    # ===== Step 1: Get skeleton grid (V2: learnable projection + joint-to-part) =====
    with torch.no_grad():
        _, feature_s = model.ctrgcn.extract_feature(skeleton)

    # Learnable pipeline (Conv1d projection + Conv1d joint-to-part)
    with torch.no_grad():
        skel_grid = model._build_skel_grid_v2(feature_s.detach())  # (1, 1, 5, 5)

    # ===== Step 2: Run RGB through ResNet stages and capture attention =====
    x = model.stem(rgb)
    x = model.layer1(x)
    x = model.layer2(x)  # (1, 512, 28, 28)

    # Extract spatial attention from cross_attn module
    with torch.no_grad():
        skel_flat = skel_grid.view(1, -1)
        ch_attn = model.cross_attn.channel_attn(skel_flat)  # (1, 512)

        ch_attn_expanded = ch_attn.unsqueeze(-1).unsqueeze(-1)
        feat_ca = x * ch_attn_expanded

        # V2: Learned upsampling via ConvTranspose2d (5x5 -> 14x14 -> 28x28)
        skel_sp = model.cross_attn.skel_upsample(skel_grid)  # (1, 1, 28, 28)

        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)
        sp_input = torch.cat([rgb_max, rgb_avg, skel_sp], dim=1)

        sp_attn = model.cross_attn.spatial_conv(sp_input)  # (1, 1, 28, 28)

    # ===== Step 3: Grad-CAM on layer4 =====
    gradcam_features = {}

    def save_activation(name):
        def hook(module, input, output):
            gradcam_features[name] = output
        return hook

    def save_gradient(name):
        def hook(module, input, output):
            gradcam_features[name + '_grad'] = output[0]
        return hook

    handle_fwd = model.layer4.register_forward_hook(save_activation('layer4'))
    handle_bwd = model.layer4.register_full_backward_hook(save_gradient('layer4'))

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
        'skel_grid': skel_grid.squeeze().cpu().numpy(),          # (5, 5)
        'skel_upsampled': skel_sp.squeeze().cpu().numpy(),       # (28, 28) learned upsample
        'spatial_attn': sp_attn.squeeze().cpu().numpy(),         # (28, 28)
        'channel_attn': ch_attn.squeeze().cpu().numpy(),         # (512,)
        'gradcam': cam,                                          # (224, 224)
        'pred_class': pred_class,
        'output_logits': output.detach().cpu().numpy().squeeze(),
    }


def plot_sample(rgb_tensor, label, attn_data, save_path, sample_name=''):
    """Create a 5-panel visualization figure."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(
        f'[V2] {sample_name}  |  GT: {LABEL_NAMES[label]}  |  Pred: {LABEL_NAMES[attn_data["pred_class"]]}',
        fontsize=14, fontweight='bold',
        color='green' if label == attn_data['pred_class'] else 'red'
    )

    # 1. Original STROI image
    img = denormalize_image(rgb_tensor)
    axes[0].imshow(img)
    axes[0].set_title('STROI Image (Input)')
    axes[0].axis('off')

    # 2. Skeleton spatiotemporal grid (5x5)
    grid = attn_data['skel_grid']
    im1 = axes[1].imshow(grid, cmap='hot', interpolation='nearest', aspect='auto')
    axes[1].set_title('Skeleton Grid (5x5)\n(Learned Conv1d proj)')
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(PART_NAMES, fontsize=8)
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels([f't{i}' for i in range(5)], fontsize=8)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # 3. Learned upsampled skeleton (28x28 from ConvTranspose2d)
    skel_up = attn_data['skel_upsampled']
    skel_up_norm = skel_up.copy()
    if skel_up_norm.max() != skel_up_norm.min():
        skel_up_norm = (skel_up_norm - skel_up_norm.min()) / (skel_up_norm.max() - skel_up_norm.min())
    im2 = axes[2].imshow(skel_up_norm, cmap='hot', interpolation='nearest')
    axes[2].set_title('Learned Upsample (28x28)\n(ConvTranspose2d)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # 4. Cross-attention spatial map (28x28 -> overlay on STROI)
    sp_attn = attn_data['spatial_attn']
    sp_attn_full = np.array(Image.fromarray((sp_attn * 255).astype(np.uint8)).resize((224, 224)))
    sp_attn_full = sp_attn_full.astype(np.float32) / 255.0
    axes[3].imshow(img)
    axes[3].imshow(sp_attn_full, cmap='jet', alpha=0.5)
    axes[3].set_title('Cross-Attention (Spatial)')
    axes[3].axis('off')

    # 5. Grad-CAM (final layer — where ResNet focuses)
    axes[4].imshow(img)
    axes[4].imshow(attn_data['gradcam'], cmap='jet', alpha=0.5)
    axes[4].set_title('Grad-CAM (ResNet Focus)')
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def log_model_weights(model):
    """Log key learnable weights of V2 model."""
    print('\n' + '=' * 60)
    print('  V2 MODEL WEIGHTS')
    print('=' * 60)

    # 1. skel_proj: Conv1d(256->1)
    proj_w = model.skel_proj[0].weight.detach().cpu()  # (1, 256, 1)
    print(f'\n  [skel_proj] Conv1d(256->1, k=1):')
    print(f'    Mean={proj_w.mean():.6f}, Std={proj_w.std():.6f}')
    print(f'    Min={proj_w.min():.6f}, Max={proj_w.max():.6f}')
    top5 = proj_w.squeeze().abs().argsort(descending=True)[:5]
    print(f'    Top-5 channels by |weight|: {["%d(%.4f)" % (i, proj_w.squeeze()[i]) for i in top5]}')

    # 2. joint_to_part: Conv1d(20->5)
    jtp_w = model.joint_to_part[0].weight.detach().cpu()  # (5, 20, 1)
    print(f'\n  [joint_to_part] Conv1d(20->5, k=1):')
    for i, name in enumerate(PART_NAMES):
        row = jtp_w[i, :, 0].numpy()
        top_joints = np.argsort(np.abs(row))[-4:][::-1]
        top_str = ', '.join(f'j{j}={row[j]:.3f}' for j in top_joints)
        print(f'    {name:8s}: sum={row.sum():.3f}, top joints: {top_str}')

    # 3. skel_upsample: ConvTranspose2d layers
    print(f'\n  [skel_upsample] ConvTranspose2d progressive 5x5->14x14->28x28:')
    for i, layer in enumerate(model.cross_attn.skel_upsample):
        if hasattr(layer, 'weight'):
            w = layer.weight.detach().cpu()
            print(f'    Layer {i} ({layer.__class__.__name__}): shape={list(w.shape)}, '
                  f'mean={w.mean():.6f}, std={w.std():.6f}')

    # 4. spatial_conv: Conv2d(3->1, 7x7)
    conv = model.cross_attn.spatial_conv[0]
    bn = model.cross_attn.spatial_conv[1]
    w = conv.weight.detach().cpu().numpy().squeeze()  # (3, 7, 7)
    channels = ['RGB MaxPool', 'RGB AvgPool', 'Skeleton (learned upsample)']
    print(f'\n  [spatial_conv] Conv2d(3->1, k=7) + BN + Sigmoid:')
    for i, c_name in enumerate(channels):
        print(f'    {c_name}: Mean={w[i].mean():.4f}, Min={w[i].min():.4f}, Max={w[i].max():.4f}')
    print(f'    BN gamma={bn.weight.item():.4f}, beta={bn.bias.item():.4f}')
    print('=' * 60)


def log_attention_details(sample_name, label, attn_data):
    """Print detailed numerical info for one sample (V2 version)."""
    grid = attn_data['skel_grid']          # (5, 5)
    skel_up = attn_data['skel_upsampled']  # (28, 28) learned upsample
    sp = attn_data['spatial_attn']         # (28, 28)
    ch = attn_data['channel_attn']         # (512,)
    logits = attn_data['output_logits']    # (10,)
    pred = attn_data['pred_class']

    print(f'\n{"~" * 60}')
    print(f'  SAMPLE: {sample_name}  |  GT: {LABEL_NAMES[label]}  |  Pred: {LABEL_NAMES[pred]}')
    print(f'{"~" * 60}')

    # 1. Skeleton grid (V2: learnable Conv1d projection + Conv1d joint-to-part)
    print(f'\n  [1] SKELETON GRID (5x5) - V2 learned projection + joint-to-part:')
    for i, name in enumerate(PART_NAMES):
        vals = '  '.join(f'{v:.3f}' for v in grid[i])
        print(f'    {name:8s}: [{vals}]')

    # 2. Learned upsampled skeleton (ConvTranspose2d 5x5 -> 28x28)
    band_h = 28 // 5
    print(f'\n  [2] LEARNED UPSAMPLE (ConvTranspose2d 5x5->28x28, per band):')
    print(f'    {"Part":8s}  {"Upsample mean":>14s}  {"Spatial attn":>14s}')
    print(f'    {"-"*8}  {"-"*14}  {"-"*14}')
    for i, name in enumerate(PART_NAMES):
        r_start = i * band_h
        r_end = min((i + 1) * band_h, 28)
        up_mean = skel_up[r_start:r_end, :].mean()
        sp_mean = sp[r_start:r_end, :].mean()
        print(f'    {name:8s}  {up_mean:14.4f}  {sp_mean:14.4f}')
    print(f'    {"":->8}  {"":->14}  {"":->14}')
    print(f'    {"Overall":8s}  {skel_up.mean():14.4f}  {sp.mean():14.4f}')
    print(f'    {"":8s}  range=[{skel_up.min():.4f}, {skel_up.max():.4f}]')
    print(f'    {"":8s}  sp range=[{sp.min():.4f}, {sp.max():.4f}]')

    # 3. Channel attention stats
    print(f'\n  [3] CHANNEL ATTENTION (512 channels):')
    print(f'    mean={ch.mean():.4f}  std={ch.std():.4f}  min={ch.min():.4f}  max={ch.max():.4f}')
    top5_idx = ch.argsort()[-5:][::-1]
    bot5_idx = ch.argsort()[:5]
    print(f'    Top 5 channels: {["%d(%.3f)" % (i, ch[i]) for i in top5_idx]}')
    print(f'    Bot 5 channels: {["%d(%.3f)" % (i, ch[i]) for i in bot5_idx]}')

    # 4. Logits
    print(f'\n  [4] OUTPUT LOGITS:')
    sorted_idx = logits.argsort()[::-1]
    for rank, i in enumerate(sorted_idx[:5]):
        marker = ' <<< GT' if i == label else ''
        marker += ' <<< PRED' if i == pred else ''
        print(f'    #{rank+1}: {LABEL_NAMES[i]:25s} = {logits[i]:+.3f}{marker}')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print('  Cross-Modal Attention Visualization — V2')
    print('  (Conv1d proj + Conv1d joint-to-part + ConvTranspose2d upsample)')
    print('=' * 60)

    # Load model
    model = load_model(args.weights, args.ctrgcn_weights, DEVICE)

    # Log model weights ONCE
    log_model_weights(model)

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
        save_path = os.path.join(args.output_dir, f'{sample_name}_attn_v2.png')
        plot_sample(rgb, label, attn_data, save_path, sample_name)

    print(f"\n[DONE] Saved {len(indices)} V2 visualizations to {args.output_dir}")


if __name__ == '__main__':
    main()
