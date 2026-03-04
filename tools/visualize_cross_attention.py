"""
Visualization: Cross-Modal Attention Heatmaps on STROI Images

Generates 3 visualizations per sample:
  1. Skeleton Grid     — raw 5×5 spatiotemporal feature grid from CTR-GCN
  2. Cross-Attention   — spatial attention map learned by CrossModalAttention
  3. Grad-CAM          — where ResNet focuses AFTER cross-attention injection

Usage:
    python tools/visualize_cross_attention.py \
        --weights ./work_dir/nucla/resnet_onthefly_v2/best_model.pt \
        --ctrgcn_weights ./result/nucla/CTROGC-GCN.pt \
        --data_path C:/Users/nguyn/Downloads/NW-UCLA-ALL/NW-UCLA-ALL \
        --rgb_path C:/ucla_stroi/ \
        --num_samples 10 \
        --output_dir ./vis_cross_attn/
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
from models.resnet_ctrgcn_ontheflyv1 import Model as CrossModalModel
from feeder.feeder_nucla_fused_ctr_resnet import Feeder

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_CLASS = 10
LABEL_NAMES = {
    0: 'Pick up with one hand', 1: 'Pick up with two hands',
    2: 'Drop trash', 3: 'Walk around', 4: 'Sit down',
    5: 'Stand up', 6: 'Donning', 7: 'Doffing',
    8: 'Throw', 9: 'Carry',
}
PART_NAMES = ['Head', 'L_hand', 'R_hand', 'L_leg', 'R_leg']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--ctrgcn_weights', type=str, default='./result/nucla/CTROGC-GCN.pt')
    parser.add_argument('--data_path', type=str, default='C:/Users/nguyn/Downloads/NW-UCLA-ALL/NW-UCLA-ALL')
    parser.add_argument('--rgb_path', type=str, default='C:/ucla_stroi/')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./vis_cross_attn/')
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
    """Run forward pass and extract intermediate attention maps + Grad-CAM.
    
    Updated for V1 model architecture:
      - Uses _build_skel_grid_v1 (learnable Conv1d projection)
      - Uses spatial_refine (residual Conv refinement) before spatial_conv
    """
    skeleton = torch.as_tensor(skeleton).unsqueeze(0).float().to(device)
    rgb = torch.as_tensor(rgb).unsqueeze(0).float().to(device)
    rgb.requires_grad = False

    # ===== Step 1: Get skeleton grid (V1: learnable skel_proj) =====
    with torch.no_grad():
        _, feature_s = model.ctrgcn.extract_feature(skeleton)  # (1, 256, 13, 20, 1)

    # V1: skel_proj is OUTSIDE no_grad so it gets gradients during training
    # For visualization we don't need gradients, but we use the same path
    with torch.no_grad():
        skel_grid = model._build_skel_grid_v1(feature_s.detach())  # (1, 1, 5, 5)

    # ===== Step 2: Run RGB through ResNet stages and capture attention =====
    x = model.stem(rgb)
    x = model.layer1(x)
    x = model.layer2(x)  # (1, 512, 28, 28)

    # Extract spatial attention from cross_attn module (V1 logic)
    with torch.no_grad():
        B, C, H, W = x.shape
        
        # Channel attention
        skel_flat = skel_grid.view(1, -1)                          # (1, 25)
        ch_attn = model.cross_attn.channel_attn(skel_flat)         # (1, 512)
        ch_attn_expanded = ch_attn.unsqueeze(-1).unsqueeze(-1)     # (1, 512, 1, 1)
        feat_ca = x * ch_attn_expanded                             # (1, 512, 28, 28)
        
        # V1: Spatial attention with learned refinement (ReLU keeps it positive)
        skel_up = F.interpolate(skel_grid, size=(H, W), mode='bilinear', align_corners=False)
        skel_sp = F.relu(skel_up + model.cross_attn.spatial_refine(skel_up))  # ★ V1 positive residual refine
        
        rgb_max = torch.max(feat_ca, dim=1, keepdim=True)[0]       # (1, 1, 28, 28)
        rgb_avg = torch.mean(feat_ca, dim=1, keepdim=True)         # (1, 1, 28, 28)
        rgb_sp_feat = torch.cat([rgb_max, rgb_avg], dim=1)         # (1, 2, 28, 28)
        
        rgb_sp_logits = model.cross_attn.rgb_spatial_conv(rgb_sp_feat)  # (1, 1, 28, 28)
        sp_attn = torch.sigmoid(rgb_sp_logits + skel_sp)           # (1, 1, 28, 28)

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

    # Capture skeleton spatial maps for comparison
    with torch.no_grad():
        sp_bilinear = F.interpolate(skel_grid, size=(28, 28), mode='bilinear', align_corners=False)
        sp_refined = F.relu(sp_bilinear + model.cross_attn.spatial_refine(sp_bilinear))  # V1 refined (kept positive)

    return {
        'skel_grid': skel_grid.squeeze().cpu().numpy(),          # (5, 5)
        'spatial_attn': sp_attn.squeeze().cpu().numpy(),         # (28, 28)
        'sp_before_proj': sp_bilinear.squeeze().cpu().numpy(),   # (28, 28) raw bilinear
        'sp_refined': sp_refined.squeeze().cpu().numpy(),        # (28, 28) after spatial_refine
        'channel_attn': ch_attn.squeeze().cpu().numpy(),         # (512,)
        'gradcam': cam,                                          # (224, 224)
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

    # Original STROI image
    img = denormalize_image(rgb_tensor)
    axes[0].imshow(img)
    axes[0].set_title('STROI Image (Input)')
    axes[0].axis('off')

    # Skeleton spatiotemporal grid (5×5)
    grid = attn_data['skel_grid']
    im1 = axes[1].imshow(grid, cmap='hot', interpolation='nearest', aspect='auto')
    axes[1].set_title('Skeleton Grid (5×5)')
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(PART_NAMES, fontsize=8)
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels([f't{i}' for i in range(5)], fontsize=8)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Cross-attention spatial map (28×28 → overlay on STROI)
    sp_attn = attn_data['spatial_attn']
    sp_attn_full = np.array(Image.fromarray((sp_attn * 255).astype(np.uint8)).resize((224, 224)))
    sp_attn_full = sp_attn_full.astype(np.float32) / 255.0
    axes[2].imshow(img)
    axes[2].imshow(sp_attn_full, cmap='jet', alpha=0.5)
    axes[2].set_title('Cross-Attention (Spatial)')
    axes[2].axis('off')

    # Grad-CAM (final layer — where ResNet focuses)
    axes[3].imshow(img)
    axes[3].imshow(attn_data['gradcam'], cmap='jet', alpha=0.5)
    axes[3].set_title('Grad-CAM (ResNet Focus)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def log_spatial_proj_weights(model):
    """Log V1 model weights: skel_proj, spatial_refine, and spatial_conv."""
    print('\n' + '=' * 60)
    print('  V1 MODEL WEIGHTS')
    print('=' * 60)
    
    # --- 1. skel_proj (Conv1d 256→1 + ReLU, learnable skeleton projection) ---
    print('\n  [1] SKEL_PROJ (Conv1d 256→1 + ReLU):')
    proj_conv = model.skel_proj[0]  # Conv1d(256, 1, 1)
    w = proj_conv.weight.detach().cpu().numpy().squeeze()  # (256,)
    print(f'    Conv1d weight: mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}')
    if proj_conv.bias is not None:
        print(f'    Conv1d bias: {proj_conv.bias.detach().cpu().item():.4f}')
    top5 = w.argsort()[-5:][::-1]
    bot5 = w.argsort()[:5]
    print(f'    Top 5 GCN channels: {["%d(%.4f)" % (i, w[i]) for i in top5]}')
    print(f'    Bot 5 GCN channels: {["%d(%.4f)" % (i, w[i]) for i in bot5]}')
    
    # --- 2. spatial_refine (2× Conv2d residual refinement) ---
    print('\n  [2] SPATIAL_REFINE (residual Conv refinement network, positive output):')
    ref_conv1 = model.cross_attn.spatial_refine[0]  # Conv2d(1, 8, 3)
    ref_bn1 = model.cross_attn.spatial_refine[1]    # BN(8)
    ref_conv2 = model.cross_attn.spatial_refine[3]  # Conv2d(8, 1, 3)
    w1 = ref_conv1.weight.detach().cpu().numpy()
    w2 = ref_conv2.weight.detach().cpu().numpy()
    print(f'    Conv2d_1 (1→8, 3×3): mean={w1.mean():.4f}, std={w1.std():.4f}')
    print(f'    BN_1 gamma={ref_bn1.weight.detach().cpu().mean().item():.4f}')
    print(f'    Conv2d_2 (8→1, 3×3): mean={w2.mean():.4f}, std={w2.std():.4f}')
    if ref_conv2.bias is not None:
        print(f'    Conv2d_2 bias: {ref_conv2.bias.detach().cpu().mean().item():.4f}')
    
    # --- 3. rgb_spatial_conv (cross-modal spatial attention base) ---
    print('\n  [3] RGB_SPATIAL_CONV (Conv2d 2→1, 7×7 + BN):')
    conv = model.cross_attn.rgb_spatial_conv[0]  # Conv2d(2, 1, 7, 7)
    bn = model.cross_attn.rgb_spatial_conv[1]    # BatchNorm2d(1)
    w = conv.weight.detach().cpu().numpy().squeeze()  # (2, 7, 7)
    channels = ["RGB MaxPool", "RGB AvgPool"]
    for i, c_name in enumerate(channels):
        print(f'    Kernel cho kênh {c_name}: mean={w[i].mean():.4f}, min={w[i].min():.4f}, max={w[i].max():.4f}')
    print(f'    BN gamma={bn.weight.detach().cpu().item():.4f}, beta={bn.bias.detach().cpu().item():.4f}')
    print(f'    BN run_mean={bn.running_mean.detach().cpu().item():.4f}, run_var={bn.running_var.detach().cpu().item():.4f}')
    
    print('\n  => V1: Skeleton grid là kết quả của Conv1d LEARNABLE (có gradient)')
    print('  => Spatial map được REFINED bởi residual Conv network trước khi cross-modal')
    print('=' * 60)


def log_attention_details(sample_name, label, attn_data):
    """Print detailed numerical info for one sample (V1 version)."""
    grid = attn_data['skel_grid']          # (5, 5)
    sp = attn_data['spatial_attn']         # (28, 28)
    sp_raw = attn_data['sp_before_proj']   # (28, 28) bilinear only
    sp_ref = attn_data['sp_refined']       # (28, 28) after spatial_refine
    ch = attn_data['channel_attn']         # (512,)
    logits = attn_data['output_logits']    # (10,)
    pred = attn_data['pred_class']
    
    print(f'\n{"~" * 60}')
    print(f'  SAMPLE: {sample_name}  |  GT: {LABEL_NAMES[label]}  |  Pred: {LABEL_NAMES[pred]}')
    print(f'{"~" * 60}')
    
    # 1. Skeleton grid (V1: from learnable Conv1d projection)
    print(f'\n  [1] SKELETON GRID (5x5) - V1 learnable skel_proj:')
    for i, name in enumerate(PART_NAMES):
        vals = '  '.join(f'{v:.3f}' for v in grid[i])
        print(f'    {name:8s}: [{vals}]')
    
    # 2. Spatial attention: Bilinear → Refined → Final
    band_h = 28 // 5
    print(f'\n  [2] SPATIAL ATTENTION (per body-part band):')
    print(f'    {"Part":8s}  {"Bilinear":>12s}  {"Refined":>12s}  {"Final":>12s}  {"Direction":>10s}')
    print(f'    {"-"*8}  {"-"*12}  {"-"*12}  {"-"*12}  {"-"*10}')
    for i, name in enumerate(PART_NAMES):
        r_start = i * band_h
        r_end = min((i + 1) * band_h, 28)
        raw_mean = sp_raw[r_start:r_end, :].mean()
        ref_mean = sp_ref[r_start:r_end, :].mean()
        final_mean = sp[r_start:r_end, :].mean()
        direction = 'SAME' if (raw_mean > sp_raw.mean()) == (final_mean > sp.mean()) else 'FLIPPED'
        print(f'    {name:8s}  {raw_mean:12.4f}  {ref_mean:12.4f}  {final_mean:12.4f}  {direction:>10s}')
    print(f'    {"":-<8}  {"":-<12}  {"":-<12}  {"":-<12}')
    print(f'    {"Overall":8s}  {sp_raw.mean():12.4f}  {sp_ref.mean():12.4f}  {sp.mean():12.4f}')
    print(f'    {"":8s}  min={sp_raw.min():.4f}   min={sp_ref.min():.4f}   min={sp.min():.4f}')
    print(f'    {"":8s}  max={sp_raw.max():.4f}   max={sp_ref.max():.4f}   max={sp.max():.4f}')
    
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
    print('  Cross-Modal Attention Visualization')
    print('=' * 60)

    # Load model
    model = load_model(args.weights, args.ctrgcn_weights, DEVICE)
    
    # Log spatial_proj weights ONCE
    log_spatial_proj_weights(model)

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
