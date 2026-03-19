"""
Gaussian Noise Robustness Experiment
=====================================
Adds Gaussian noise to skeleton JOINT COORDINATES (not GCN features) at
different sigma levels to simulate real-world skeleton estimation noise.

This tests how robust the cross-modal attention is when skeleton joint
positions are imprecise (e.g., from noisy depth sensors, occluded joints).

Noise is applied to raw skeleton input x_s (B, 3, T, V, M) before GCN:
  x_s_noisy = x_s + N(0, sigma^2)

Experiments at sigma = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

Outputs:
  - Accuracy vs Noise Level curve
  - Per-class robustness analysis
  - Degradation rate analysis

Usage:
    python tools/gaussian_noise_experiment.py \
        --model_version v2 \
        --weights path/to/model.pt \
        --data_path C:/Users/nguyn/Downloads/NW-UCLA-ALL/NW-UCLA-ALL \
        --rgb_path C:/Users/nguyn/Downloads/ucla_fivefs-20260302T172922Z-1-001/ucla_fivefs/
"""
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm

sys.path.append(os.getcwd())

from models.ctrgcn import Model as CTR_GCN_Model
from feeder.feeder_nucla_fused_ctr_resnet import Feeder as FusedFeeder

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_CLASS = 10
LABEL_NAMES = {
    0: 'Pick up (1 hand)', 1: 'Pick up (2 hands)',
    2: 'Drop trash', 3: 'Walk around', 4: 'Sit down',
    5: 'Stand up', 6: 'Donning', 7: 'Doffing',
    8: 'Throw', 9: 'Carry',
}
CTRGCN_WEIGHTS = './result/nucla/CTROGC-GCN.pt'

SIGMA_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


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


def load_model(model_version, weights_path):
    """Load the specified model version with CTR-GCN injected."""
    if model_version == 'v0':
        from models.resnet_ctrgcn_onthefly import Model as ModelClass
    elif model_version == 'v2':
        from models.resnet_ctrgcn_ontheflyv2 import Model as ModelClass
    elif model_version == 'v7':
        from models.resnet_ctrgcn_ontheflyv7 import Model as ModelClass
    elif model_version == 'v8':
        from models.resnet_ctrgcn_ontheflyv8 import Model as ModelClass
    elif model_version == 'v9':
        from models.resnet_ctrgcn_ontheflyv9 import Model as ModelClass
    elif model_version == 'v10':
        from models.resnet_ctrgcn_ontheflyv10 import Model as ModelClass
    elif model_version == 'v11':
        from models.resnet_ctrgcn_ontheflyv11 import Model as ModelClass
    elif model_version == 'v12':
        from models.resnet_ctrgcn_ontheflyv12 import Model as ModelClass
    elif model_version == 'v13':
        from models.resnet_ctrgcn_ontheflyv13 import Model as ModelClass
    elif model_version == 'v14':
        from models.resnet_ctrgcn_ontheflyv14 import Model as ModelClass
    elif model_version == 'v17':
        from models.resnet_ctrgcn_ontheflyv17 import Model as ModelClass
    else:
        raise ValueError(f'Unknown model version: {model_version}')

    model = ModelClass(num_class=NUM_CLASS, pretrained=False).to(DEVICE)

    # Inject CTR-GCN
    ctrgcn = CTR_GCN_Model(
        num_class=NUM_CLASS, num_point=20, num_person=1,
        graph='graph.ucla.Graph', graph_args={'labeling_mode': 'spatial'},
    ).to(DEVICE)
    ctrgcn_state = load_weights_robust(CTRGCN_WEIGHTS, DEVICE)
    ctrgcn.load_state_dict(ctrgcn_state, strict=False)
    ctrgcn.eval()
    for p in ctrgcn.parameters():
        p.requires_grad = False
    model.ctrgcn = ctrgcn

    # Load trained weights
    state_dict = load_weights_robust(weights_path, DEVICE)
    model_state = model.state_dict()
    for k, v in state_dict.items():
        if k in model_state and model_state[k].size() == v.size():
            model_state[k] = v
    model.load_state_dict(model_state)
    model.eval()
    return model


def evaluate_with_noise(model, dataloader, sigma=0.0, n_trials=3):
    """Evaluate model with Gaussian noise added to skeleton coordinates.

    Noise is added to raw skeleton data x_s BEFORE it enters the frozen GCN.
    For sigma > 0, we run n_trials and average the accuracy (Monte Carlo).

    Args:
        sigma: std of Gaussian noise added to skeleton coordinates
        n_trials: number of random trials to average over (for sigma > 0)
    """
    if sigma == 0.0:
        n_trials = 1

    trial_accuracies = []
    trial_per_class = []

    for trial in range(n_trials):
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data, rgb, label in tqdm(dataloader,
                                         desc=f'  sigma={sigma:.3f} trial={trial+1}/{n_trials}',
                                         leave=False):
                data = data.float().to(DEVICE)
                rgb = rgb.float().to(DEVICE)

                # ADD GAUSSIAN NOISE to skeleton coordinates
                if sigma > 0:
                    noise = torch.randn_like(data) * sigma
                    data_noisy = data + noise
                else:
                    data_noisy = data

                output = model(data_noisy, rgb)
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(label.numpy())

        preds = np.array(all_preds)
        labels = np.array(all_labels)
        acc = (preds == labels).mean() * 100
        trial_accuracies.append(acc)

        # Per-class
        per_class = {}
        for c in range(NUM_CLASS):
            mask = (labels == c)
            if mask.sum() > 0:
                per_class[c] = (preds[mask] == c).sum() / mask.sum() * 100
            else:
                per_class[c] = 0.0
        trial_per_class.append(per_class)

    # Average across trials
    mean_acc = np.mean(trial_accuracies)
    std_acc = np.std(trial_accuracies) if n_trials > 1 else 0.0

    mean_per_class = {}
    for c in range(NUM_CLASS):
        vals = [pc[c] for pc in trial_per_class]
        mean_per_class[c] = np.mean(vals)

    return mean_acc, std_acc, mean_per_class


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=str, default='v2', choices=['v0', 'v2', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v17'])
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--rgb_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./gaussian_noise_results')
    parser.add_argument('--n_trials', type=int, default=3, help='Monte Carlo trials per sigma level')
    parser.add_argument('--sigmas', type=float, nargs='+', default=SIGMA_LEVELS,
                        help='Noise sigma levels to test')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 70)
    print(f'  GAUSSIAN NOISE ROBUSTNESS: {args.model_version.upper()}')
    print('=' * 70)

    # Load dataset
    print('\n[1/3] Loading dataset...')
    dataset = FusedFeeder(
        data_path=args.data_path, label_path='val',
        rgb_path=args.rgb_path, temporal_rgb_frames=5,
        random_choose=False, random_shift=False, random_move=False,
        window_size=52, normalization=False,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f'  Test samples: {len(dataset)}')

    # Load model
    print(f'\n[2/3] Loading model ({args.model_version})...')
    model = load_model(args.model_version, args.weights)

    # Evaluate at each noise level
    print(f'\n[3/3] Evaluating at {len(args.sigmas)} noise levels (x{args.n_trials} trials)...')
    results = {}
    for sigma in args.sigmas:
        mean_acc, std_acc, per_class = evaluate_with_noise(
            model, loader, sigma=sigma, n_trials=args.n_trials
        )
        results[sigma] = {
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'per_class': per_class,
        }
        print(f'  sigma={sigma:.3f}: {mean_acc:.2f}% (+/- {std_acc:.2f}%)')

    # ========================================================
    # Generate outputs
    # ========================================================
    sigmas = sorted(results.keys())
    accuracies = [results[s]['mean_acc'] for s in sigmas]
    stds = [results[s]['std_acc'] for s in sigmas]

    # Report
    report = []
    report.append('=' * 70)
    report.append(f'  GAUSSIAN NOISE ROBUSTNESS: {args.model_version.upper()}')
    report.append('=' * 70)
    report.append(f'\n  Model: {args.model_version}')
    report.append(f'  Weights: {args.weights}')
    report.append(f'  Trials per sigma: {args.n_trials}')
    report.append(f'\n  {"Sigma":<10s} {"Accuracy":<12s} {"Std":<10s} {"Delta":<10s}')
    report.append(f'  {"-"*42}')

    base_acc = results[0.0]['mean_acc'] if 0.0 in results else accuracies[0]
    for sigma in sigmas:
        r = results[sigma]
        delta = r['mean_acc'] - base_acc
        report.append(f'  {sigma:<10.3f} {r["mean_acc"]:<12.2f} {r["std_acc"]:<10.2f} {delta:+<10.2f}')

    # Per-class robustness (which classes degrade fastest?)
    report.append(f'\n{"="*70}')
    report.append('  PER-CLASS DEGRADATION (accuracy at max noise vs clean)')
    report.append(f'{"="*70}')

    max_sigma = max(sigmas)
    clean = results[0.0]['per_class'] if 0.0 in results else results[sigmas[0]]['per_class']
    noisy = results[max_sigma]['per_class']

    report.append(f'\n  {"Class":<22s} {"Clean":<10s} {"sigma={:.2f}":<10s} {"Delta":<10s}'.format(max_sigma))
    report.append(f'  {"-"*52}')
    degradations = []
    for c in range(NUM_CLASS):
        d = noisy[c] - clean[c]
        degradations.append((c, d))
        report.append(f'  {LABEL_NAMES[c]:<22s} {clean[c]:<10.1f} {noisy[c]:<10.1f} {d:+<10.1f}')

    # Most fragile classes
    degradations.sort(key=lambda x: x[1])
    report.append(f'\n  Most fragile classes (largest degradation):')
    for c, d in degradations[:3]:
        report.append(f'    {LABEL_NAMES[c]}: {d:+.1f}%')

    report_text = '\n'.join(report)
    print(report_text)

    report_path = os.path.join(args.output_dir, 'gaussian_noise_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f'\n  Report saved: {report_path}')

    # ========================================================
    # Plots
    # ========================================================
    # 1. Accuracy vs Noise Level
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(sigmas, accuracies, yerr=stds, marker='o', linewidth=2,
                capsize=5, color='#2196F3', markersize=8)
    ax.fill_between(sigmas,
                    [a - s for a, s in zip(accuracies, stds)],
                    [a + s for a, s in zip(accuracies, stds)],
                    alpha=0.2, color='#2196F3')
    ax.set_xlabel('Gaussian Noise Sigma (σ)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title(f'Robustness to Joint Coordinate Noise ({args.model_version.upper()})',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

    # Add annotation for degradation
    if len(sigmas) >= 2:
        drop = base_acc - accuracies[-1]
        ax.annotate(f'Total drop: -{drop:.1f}%',
                    xy=(sigmas[-1], accuracies[-1]),
                    xytext=(sigmas[-1] - 0.1, accuracies[-1] + 8),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=11, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'accuracy_vs_noise.png'), dpi=200)
    plt.close()
    print('  Figure saved: accuracy_vs_noise.png')

    # 2. Per-class heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    heatmap_data = np.zeros((NUM_CLASS, len(sigmas)))
    for j, sigma in enumerate(sigmas):
        for c in range(NUM_CLASS):
            heatmap_data[c, j] = results[sigma]['per_class'][c]

    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f'{s:.3f}' for s in sigmas])
    ax.set_yticks(range(NUM_CLASS))
    ax.set_yticklabels([LABEL_NAMES[c][:15] for c in range(NUM_CLASS)], fontsize=9)
    ax.set_xlabel('Noise Sigma (σ)')
    ax.set_title('Per-Class Accuracy vs Noise Level', fontweight='bold')

    for i in range(NUM_CLASS):
        for j in range(len(sigmas)):
            val = heatmap_data[i, j]
            color = 'white' if val < 40 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7, color=color)

    plt.colorbar(im, label='Accuracy (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'perclass_noise_heatmap.png'), dpi=200)
    plt.close()
    print('  Figure saved: perclass_noise_heatmap.png')

    print(f'\n  All outputs in: {os.path.abspath(args.output_dir)}/')
    print('=' * 70)


if __name__ == '__main__':
    main()
