"""NTU-60 V17 Full Evaluation Script
Chạy tất cả: Attention Analysis, Gaussian Noise, Error Analysis, Ensemble
Lưu mọi kết quả vào 1 folder duy nhất.

Usage:
    python tools/evaluate_ntu_v17.py \\
        --onthefly_weights C:/Users/nguyn/Downloads/ontheflyv17_ntu_cv.pt \\
        --ctrgcn_weights ./result/ntu/ntu_gcn_xview.pt \\
        --npz_path <path>/NTU60_CV_CLEAN.npz \\
        --rgb_path <path>/ntu_rgbd_fivefs.zip \\
        --output_dir ./results/ntu_v17_eval
"""
import sys, os, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm
from scipy.special import softmax

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.getcwd())

from models.ctrgcn import Model as CTR_GCN_Model
from feeder.feeder_ntu_fused_ctr_resnet import Feeder

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_CLASS = 60


# ─────────────────────────── ARG PARSE ───────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='NTU-60 V17 Full Evaluation')
    p.add_argument('--onthefly_weights', type=str, required=True)
    p.add_argument('--ctrgcn_weights',   type=str, default='./result/ntu/ntu_gcn_xview.pt')
    p.add_argument('--npz_path',         type=str, required=True)
    p.add_argument('--rgb_path',         type=str, required=True)
    p.add_argument('--split',            type=str, default='CV', choices=['CV', 'CS'])
    p.add_argument('--output_dir',       type=str, default='./results/ntu_v17_eval')
    p.add_argument('--batch_size',       type=int, default=64)
    p.add_argument('--num_workers',      type=int, default=4)
    # Gaussian noise
    p.add_argument('--sigmas', nargs='+', type=float,
                   default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5])
    p.add_argument('--n_trials', type=int, default=1,
                   help='Monte Carlo trials per sigma (1 đủ cho NTU vì val set lớn)')
    return p.parse_args()


# ─────────────────────────── LOAD UTILS ──────────────────────────

def load_weights_robust(path, device):
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if key in raw:
                raw = raw[key]
                break
    new_state = OrderedDict()
    for k, v in raw.items():
        new_state[k.replace('module.', '')] = v
    return new_state


def load_ctrgcn(weights_path, device):
    model = CTR_GCN_Model(
        num_class=NUM_CLASS, num_point=25, num_person=2,
        graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'},
    ).to(device)
    model.load_state_dict(load_weights_robust(weights_path, device), strict=False)
    model.eval()
    print(f'  [OK] CTR-GCN: {weights_path}')
    return model


def load_onthefly(weights_path, ctrgcn_path, device):
    import importlib
    mod = importlib.import_module('models.resnet_ctrgcn_ontheflyv17')
    model = mod.Model(
        num_class=NUM_CLASS, pretrained=False,
        consistency_weight=0.1, sp_feat_channels=4,
        init_sigma_sharp=3.0, init_sigma_coarse=8.0,
        init_temp_sharp=0.3, init_temp_coarse=1.5,
        num_point=25, num_person=2,
    ).to(device)

    # Inject frozen CTR-GCN
    ctrgcn = CTR_GCN_Model(
        num_class=NUM_CLASS, num_point=25, num_person=2,
        graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'},
    ).to(device)
    ctrgcn.load_state_dict(load_weights_robust(ctrgcn_path, device), strict=False)
    ctrgcn.eval()
    for p in ctrgcn.parameters():
        p.requires_grad = False
    model.ctrgcn = ctrgcn

    sd = load_weights_robust(weights_path, device)
    ms = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in ms and ms[k].size() == v.size():
            ms[k] = v
            loaded += 1
    model.load_state_dict(ms)
    model.eval()
    print(f'  [OK] OnTheFly V17 ({loaded} tensors): {weights_path}')
    return model


def make_loader(args, pin_memory=True):
    ds = Feeder(
        npz_path=args.npz_path,
        rgb_path=args.rgb_path,
        label_path='val',
        split=args.split,
        time_steps=64,
        temporal_rgb_frames=5,
        random_flip=False,
        random_choose=False,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    ), ds


# ─────────────────────── BASE INFERENCE ──────────────────────────
# Chạy 1 lần, thu thập: scores, labels, attention maps (hooks)

def run_base_inference(ctrgcn_model, onthefly_model, loader):
    """Thu thập logits + attention maps trong 1 lần forward."""
    captured = {}
    handles = []

    # Hook 1: spatial_gate output → (B, 1, H, W) gate [0,1]
    def hook_gate(module, inp, out):
        captured.setdefault('gate', []).append(out.detach().cpu())

    # Hook 2: channel_attn output → (B, C, 1, 1)
    def hook_ch(module, inp, out):
        captured.setdefault('ch_attn', []).append(out.detach().cpu())

    # Hook 3: spatial_net_sharp output (logits, pre-sigmoid)
    def hook_sharp(module, inp, out):
        T_s = onthefly_model.cross_attn.log_temp_sharp.exp().detach()
        attn = torch.sigmoid(out.detach() / T_s)
        captured.setdefault('sp_sharp', []).append(attn.cpu())

    handles.append(onthefly_model.cross_attn.spatial_gate.register_forward_hook(hook_gate))
    handles.append(onthefly_model.cross_attn.channel_attn.register_forward_hook(hook_ch))
    handles.append(onthefly_model.cross_attn.spatial_net_sharp.register_forward_hook(hook_sharp))

    gcn_scores, v17_scores, all_labels = [], [], []

    with torch.no_grad():
        for skeleton, rgb, label in tqdm(loader, desc='Base inference'):
            skeleton = skeleton.float().to(DEVICE)
            rgb      = rgb.float().to(DEVICE)
            gcn_scores.append(ctrgcn_model(skeleton).cpu().numpy())
            v17_scores.append(onthefly_model(skeleton, rgb).cpu().numpy())
            lbl = label.numpy() if isinstance(label, torch.Tensor) else np.array(label)
            all_labels.append(lbl)

    for h in handles:
        h.remove()

    gcn_scores = np.concatenate(gcn_scores)
    v17_scores = np.concatenate(v17_scores)
    labels     = np.concatenate(all_labels)

    attn_data = {
        'gate':     torch.cat(captured.get('gate',    []), dim=0),  # (N, 1, H, W)
        'sp_sharp': torch.cat(captured.get('sp_sharp',[]), dim=0),  # (N, 1, H, W)
        'ch_attn':  torch.cat(captured.get('ch_attn', []), dim=0),  # (N, C, 1, 1)
    }
    return gcn_scores, v17_scores, labels, attn_data


# ─────────────────── 1. ATTENTION ANALYSIS ───────────────────────

def compute_entropy(arr2d):
    """Spatial entropy of a 2D map (flattened, softmax-normalized)."""
    flat = arr2d.flatten().astype(np.float64)
    flat = flat / (flat.sum() + 1e-9)
    flat = flat[flat > 1e-12]
    return float(-np.sum(flat * np.log(flat)))


def analyze_attention(attn_data, labels, output_dir):
    print('\n[1/4] Attention Map Analysis...')
    gate    = attn_data['gate'].squeeze(1).numpy()     # (N, H, W)
    sharp   = attn_data['sp_sharp'].squeeze(1).numpy() # (N, H, W)
    ch_attn = attn_data['ch_attn'].squeeze(-1).squeeze(-1).numpy()  # (N, C)

    # Combined spatial = sharp * gate (same as model)
    combined = gate * sharp  # (N, H, W)

    H, W = combined.shape[1], combined.shape[2]
    uniform_entropy = float(np.log(H * W))

    # Per-sample metrics
    entropies, variances, peak_means = [], [], []
    for m in combined:
        entropies.append(compute_entropy(m))
        variances.append(float(m.var()))
        mean_v = m.mean()
        peak_means.append(float(m.max() / (mean_v + 1e-9)))

    # Per-class average attention
    class_attn  = {}
    class_sharp = {}
    class_gate  = {}
    class_vecs  = {}  # for inter-class similarity
    for c in range(NUM_CLASS):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        class_attn[c]  = combined[idx].mean(axis=0)
        class_sharp[c] = sharp[idx].mean(axis=0)
        class_gate[c]  = gate[idx].mean(axis=0)
        class_vecs[c]  = combined[idx].reshape(len(idx), -1).mean(axis=0)

    # Intra / inter-class similarity
    classes = sorted(class_vecs.keys())
    n_c = len(classes)
    sim_matrix = np.zeros((n_c, n_c))
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            vi, vj = class_vecs[ci], class_vecs[cj]
            ni = np.linalg.norm(vi); nj = np.linalg.norm(vj)
            if ni > 0 and nj > 0:
                sim_matrix[i, j] = float(np.dot(vi, vj) / (ni * nj))

    intra = float(np.diag(sim_matrix).mean())
    mask  = ~np.eye(n_c, dtype=bool)
    inter = float(sim_matrix[mask].mean())
    disc  = intra - inter

    # ── REPORT ──
    lines = [
        '=' * 60,
        '  ATTENTION MAP ANALYSIS — NTU-60 V17',
        '=' * 60,
        f'  Map size          : {H}x{W}',
        f'  Uniform entropy   : {uniform_entropy:.3f}',
        '',
        '  OVERALL METRICS (combined = sharp * gate):',
        f'    Spatial Entropy   : {np.mean(entropies):.3f} ± {np.std(entropies):.3f}'
        f'  (uniform={uniform_entropy:.3f})',
        f'    Spatial Variance  : {np.mean(variances):.4f} ± {np.std(variances):.4f}',
        f'    Peak-to-Mean Ratio: {np.mean(peak_means):.2f} ± {np.std(peak_means):.2f}',
        '',
        '  INTER-CLASS SIMILARITY:',
        f'    Intra-class sim   : {intra:.4f}',
        f'    Inter-class sim   : {inter:.4f}',
        f'    Discrimination Gap: {disc:.4f}  '
        + ('(GOOD > 0)' if disc > 0 else '(BAD < 0)'),
        '=' * 60,
    ]
    report_path = os.path.join(output_dir, 'attention_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))

    # ── PLOT 1: Per-class average attention (10x6 grid) ──
    n_cols = 10
    n_rows = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    for idx, c in enumerate(classes[:n_rows * n_cols]):
        r, col = divmod(idx, n_cols)
        ax = axes[r, col]
        ax.imshow(class_attn[c], cmap='hot', vmin=0)
        ax.set_title(f'C{c}', fontsize=7)
        ax.axis('off')
    for idx in range(len(classes), n_rows * n_cols):
        r, col = divmod(idx, n_cols)
        axes[r, col].axis('off')
    fig.suptitle('Average Spatial Attention per Class (V17 sharp×gate)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_perclass.png'), dpi=100)
    plt.close()

    # ── PLOT 2: Inter-class similarity matrix ──
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'Inter-Class Attention Similarity (Disc Gap={disc:.3f})', fontsize=12)
    ax.set_xlabel('Class'); ax.set_ylabel('Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_similarity.png'), dpi=100)
    plt.close()

    # ── PLOT 3: Metric distributions ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(entropies, bins=40, color='steelblue', edgecolor='k', alpha=0.8)
    axes[0].axvline(uniform_entropy, color='red', linestyle='--', label=f'Uniform={uniform_entropy:.2f}')
    axes[0].set_title('Spatial Entropy'); axes[0].legend()
    axes[1].hist(variances, bins=40, color='salmon', edgecolor='k', alpha=0.8)
    axes[1].set_title('Spatial Variance')
    axes[2].hist(peak_means, bins=40, color='mediumseagreen', edgecolor='k', alpha=0.8)
    axes[2].set_title('Peak-to-Mean Ratio')
    plt.suptitle('Attention Metric Distributions — NTU-60 V17', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_metrics.png'), dpi=100)
    plt.close()

    print(f'  → Saved: attention_report.txt, attention_perclass.png, '
          f'attention_similarity.png, attention_metrics.png')
    return {'entropy': np.mean(entropies), 'variance': np.mean(variances),
            'peak_mean': np.mean(peak_means), 'disc_gap': disc,
            'uniform_entropy': uniform_entropy}


# ─────────────────── 2. GAUSSIAN NOISE ───────────────────────────

def run_gaussian_noise(onthefly_model, dataset, args, output_dir):
    print('\n[2/4] Gaussian Noise Robustness...')
    sigmas  = args.sigmas
    results = {}  # sigma -> mean_acc

    for sigma in sigmas:
        n_trials = 1 if sigma == 0.0 else args.n_trials
        trial_accs = []
        for trial in range(n_trials):
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
            all_preds, all_labels = [], []
            with torch.no_grad():
                for skeleton, rgb, label in tqdm(
                    loader, desc=f'  σ={sigma:.3f} trial={trial+1}/{n_trials}', leave=False
                ):
                    skeleton = skeleton.float().to(DEVICE)
                    rgb      = rgb.float().to(DEVICE)
                    if sigma > 0:
                        skeleton = skeleton + torch.randn_like(skeleton) * sigma
                    out = onthefly_model(skeleton, rgb)
                    all_preds.extend(out.argmax(dim=1).cpu().numpy())
                    all_labels.extend(
                        label.numpy() if isinstance(label, torch.Tensor) else np.array(label)
                    )
            preds  = np.array(all_preds)
            lbl    = np.array(all_labels)
            acc    = (preds == lbl).mean() * 100
            trial_accs.append(acc)
        mean_acc = np.mean(trial_accs)
        results[sigma] = mean_acc
        print(f'    σ={sigma:.3f} → {mean_acc:.2f}%')

    baseline = results[0.0]
    lines = [
        '=' * 55,
        '  GAUSSIAN NOISE ROBUSTNESS — NTU-60 V17',
        '=' * 55,
        f'  {"Sigma":<8} {"Accuracy (%)":>14} {"Delta":>10}',
        f'  {"-"*35}',
    ]
    for s in sigmas:
        delta = results[s] - baseline
        lines.append(f'  {s:<8.3f} {results[s]:>14.2f} {delta:>+10.2f}%')
    lines += [
        '=' * 55,
        f'  Total drop (σ=0 → σ={sigmas[-1]:.2f}): '
        f'{results[sigmas[-1]] - baseline:+.2f}%',
    ]
    report_path = os.path.join(output_dir, 'noise_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))

    # ── PLOT ──
    xs   = sigmas
    ys   = [results[s] for s in xs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, 'o-', color='royalblue', linewidth=2, markersize=8)
    ax.fill_between(xs, ys, alpha=0.15, color='royalblue')
    ax.axhline(baseline, color='gray', linestyle='--', alpha=0.6, label=f'Clean={baseline:.2f}%')
    ax.set_xlabel('Gaussian Noise σ', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Robustness to Skeleton Noise — NTU-60 V17', fontsize=13)
    ax.legend(); ax.grid(alpha=0.3)
    total_drop = ys[-1] - baseline
    ax.annotate(f'Δ={total_drop:+.1f}%', xy=(xs[-1], ys[-1]),
                xytext=(-40, 10), textcoords='offset points',
                fontsize=10, color='red')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'noise_accuracy.png'), dpi=100)
    plt.close()

    print(f'  → Saved: noise_report.txt, noise_accuracy.png')
    return results


# ─────────────────── 3. ERROR ANALYSIS ───────────────────────────

def run_error_analysis(gcn_scores, v17_scores, labels, output_dir):
    print('\n[3/4] Error Analysis: CTR-GCN vs V17...')

    gcn_preds = np.argmax(gcn_scores, axis=1)
    v17_preds = np.argmax(v17_scores, axis=1)
    gcn_probs = softmax(gcn_scores, axis=1)
    v17_probs = softmax(v17_scores, axis=1)
    gcn_confs = gcn_probs.max(axis=1)
    v17_confs = v17_probs.max(axis=1)

    gcn_correct = (gcn_preds == labels)
    v17_correct = (v17_preds == labels)

    both_right  = (gcn_correct  & v17_correct).sum()
    improved    = (~gcn_correct & v17_correct).sum()   # V17 fixed GCN's errors
    blind_spots = (gcn_correct  & ~v17_correct).sum()  # V17 broke GCN's correct
    both_wrong  = (~gcn_correct & ~v17_correct).sum()
    N = len(labels)

    acc_gcn = gcn_correct.mean() * 100
    acc_v17 = v17_correct.mean() * 100

    # Per-class
    per_class = {}
    for c in range(NUM_CLASS):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        imp  = (~gcn_correct[idx] & v17_correct[idx]).sum()
        bsp  = (gcn_correct[idx]  & ~v17_correct[idx]).sum()
        per_class[c] = {'improved': int(imp), 'blind': int(bsp), 'net': int(imp - bsp)}

    # Confidence of blind spots
    blind_idx = np.where(gcn_correct & ~v17_correct)[0]
    blind_conf_mean = float(v17_confs[blind_idx].mean()) if len(blind_idx) else 0.0
    correct_conf_mean = float(v17_confs[v17_correct].mean())

    # Top confusion pairs in blind spots
    from collections import Counter
    conf_pairs = Counter()
    for i in blind_idx:
        conf_pairs[(int(labels[i]), int(v17_preds[i]))] += 1
    top_confusions = conf_pairs.most_common(10)

    lines = [
        '=' * 65,
        '  ERROR ANALYSIS: CTR-GCN vs OnTheFly V17 — NTU-60',
        '=' * 65,
        f'  CTR-GCN Accuracy : {acc_gcn:.2f}%',
        f'  V17     Accuracy : {acc_v17:.2f}%',
        f'  Improvement      : {acc_v17 - acc_gcn:+.2f}%',
        '',
        f'  {"Category":<22} {"Count":>8} {"(%)":>8}',
        f'  {"-"*40}',
        f'  {"Both Correct":<22} {both_right:>8} {both_right/N*100:>7.1f}%',
        f'  {"V17 Improved (GCN→V17)":<22} {improved:>8} {improved/N*100:>7.1f}%',
        f'  {"Blind Spots (V17<GCN)":<22} {blind_spots:>8} {blind_spots/N*100:>7.1f}%',
        f'  {"Both Wrong":<22} {both_wrong:>8} {both_wrong/N*100:>7.1f}%',
        '',
        f'  Net gain vs GCN  : +{improved} - {blind_spots} = {improved-blind_spots:+d} samples',
        '',
        '  CONFIDENCE (V17):',
        f'    When V17 correct  : {correct_conf_mean:.4f}',
        f'    When V17 blind    : {blind_conf_mean:.4f}',
        f'    Gap               : {correct_conf_mean - blind_conf_mean:.4f}'
        + (' (model unsure when wrong ✓)' if correct_conf_mean - blind_conf_mean > 0.1 else ''),
        '',
        '  TOP-10 CONFUSION PAIRS IN BLIND SPOTS:',
        f'  {"True→Pred":<25} {"Count":>6}',
        f'  {"-"*35}',
    ]
    for (true_c, pred_c), cnt in top_confusions:
        lines.append(f'  Class {true_c:2d} → Class {pred_c:2d}          {cnt:>6}')

    lines += [
        '',
        '  PER-CLASS NET (top 10 gained / top 10 lost):',
    ]
    sorted_net = sorted(per_class.items(), key=lambda x: x[1]['net'], reverse=True)
    lines.append(f'  {"Class":>6} {"Improved":>10} {"Blind":>8} {"Net":>6}')
    lines.append(f'  {"-"*35}')
    for c, d in sorted_net[:10]:
        lines.append(f'  {c:>6} {d["improved"]:>10} {d["blind"]:>8} {d["net"]:>+6}')
    lines.append('  ...(worst)...')
    for c, d in sorted_net[-10:]:
        lines.append(f'  {c:>6} {d["improved"]:>10} {d["blind"]:>8} {d["net"]:>+6}')
    lines.append('=' * 65)

    report_path = os.path.join(output_dir, 'error_analysis.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))

    # ── PLOT 1: 4-case bar ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cats   = ['Both Correct', 'V17 Improved', 'Blind Spots', 'Both Wrong']
    counts = [both_right, improved, blind_spots, both_wrong]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
    axes[0].bar(cats, counts, color=colors, edgecolor='k', alpha=0.85)
    for i, (c, v) in enumerate(zip(cats, counts)):
        axes[0].text(i, v + N * 0.002, f'{v}\n({v/N*100:.1f}%)',
                     ha='center', va='bottom', fontsize=9)
    axes[0].set_title('CTR-GCN vs V17 — Sample Outcomes')
    axes[0].set_ylabel('# Samples')

    # Per-class net improvement bar
    net_vals = [per_class[c]['net'] for c in range(NUM_CLASS) if c in per_class]
    net_cols = ['#3498db' if v >= 0 else '#e74c3c' for v in net_vals]
    axes[1].bar(range(len(net_vals)), net_vals, color=net_cols, edgecolor='none', alpha=0.8)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Class'); axes[1].set_ylabel('Net Improvement')
    axes[1].set_title('Per-Class Net: V17 vs CTR-GCN')

    plt.suptitle(f'Error Analysis — NTU-60 V17 (acc={acc_v17:.2f}% vs GCN {acc_gcn:.2f}%)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=100)
    plt.close()

    # ── PLOT 2: Confidence histogram ──
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(v17_confs[v17_correct], bins=40, alpha=0.6, color='#2ecc71',
            label=f'Correct (μ={correct_conf_mean:.3f})', density=True)
    ax.hist(v17_confs[~v17_correct], bins=40, alpha=0.6, color='#e74c3c',
            label=f'Wrong/Blind (μ={v17_confs[~v17_correct].mean():.3f})', density=True)
    ax.set_xlabel('V17 Confidence'); ax.set_ylabel('Density')
    ax.set_title('Confidence Distribution: Correct vs Wrong (V17)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_hist.png'), dpi=100)
    plt.close()

    print(f'  → Saved: error_analysis.txt, error_analysis.png, confidence_hist.png')
    return {'acc_gcn': acc_gcn, 'acc_v17': acc_v17, 'improved': int(improved),
            'blind_spots': int(blind_spots)}


# ─────────────────────── 4. ENSEMBLE ─────────────────────────────

def run_ensemble(gcn_scores, v17_scores, labels, output_dir):
    print('\n[4/4] Ensemble Alpha Sweep...')
    gcn_norm = softmax(gcn_scores, axis=1)
    v17_norm = softmax(v17_scores, axis=1)

    acc_gcn = (np.argmax(gcn_norm, axis=1) == labels).mean() * 100
    acc_v17 = (np.argmax(v17_norm, axis=1) == labels).mean() * 100

    alphas     = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0]
    alpha_accs = {}
    for a in alphas:
        combo = v17_norm + a * gcn_norm
        acc   = (np.argmax(combo, axis=1) == labels).mean() * 100
        alpha_accs[a] = acc

    best_alpha = max(alpha_accs, key=alpha_accs.get)
    best_acc   = alpha_accs[best_alpha]

    lines = [
        '=' * 55,
        '  ENSEMBLE — OnTheFly V17 + α × CTR-GCN (NTU-60)',
        '=' * 55,
        f'  CTR-GCN Only : {acc_gcn:.2f}%',
        f'  V17 Only     : {acc_v17:.2f}%',
        '',
        f'  {"Alpha":<8} {"Ensemble Acc (%)":>18} {"vs Best Single":>16}',
        f'  {"-"*45}',
    ]
    best_single = max(acc_gcn, acc_v17)
    for a in alphas:
        delta  = alpha_accs[a] - best_single
        marker = ' ← BEST' if a == best_alpha else ''
        lines.append(f'  {a:<8.1f} {alpha_accs[a]:>18.2f} {delta:>+15.2f}%{marker}')
    lines += [
        '=' * 55,
        f'  Best Ensemble : {best_acc:.2f}% (alpha={best_alpha})',
        f'  Improvement   : {best_acc - best_single:+.2f}% over best single model',
        '=' * 55,
    ]
    report_path = os.path.join(output_dir, 'ensemble_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar: single vs ensemble
    names = ['CTR-GCN', 'V17', f'Ensemble\n(α={best_alpha})']
    accs  = [acc_gcn, acc_v17, best_acc]
    cols  = ['#3498db', '#e67e22', '#2ecc71']
    bars  = axes[0].bar(names, accs, color=cols, edgecolor='k', alpha=0.85, width=0.5)
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                     f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].set_ylim(min(accs) - 2, max(accs) + 2)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Single vs Ensemble Accuracy')

    # Line: alpha sweep
    xs = alphas
    ys = [alpha_accs[a] for a in xs]
    axes[1].plot(xs, ys, 'o-', color='purple', linewidth=2, markersize=8)
    axes[1].axhline(best_single, color='gray', linestyle='--', alpha=0.7,
                    label=f'Best single={best_single:.2f}%')
    axes[1].scatter([best_alpha], [best_acc], color='red', s=120, zorder=5,
                    label=f'Best α={best_alpha} → {best_acc:.2f}%')
    axes[1].set_xlabel('Alpha (CTR-GCN weight)', fontsize=12)
    axes[1].set_ylabel('Ensemble Accuracy (%)', fontsize=12)
    axes[1].set_title('Alpha Sweep — V17 + α×CTR-GCN')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle('Ensemble Evaluation — NTU-60 CV', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble.png'), dpi=100)
    plt.close()

    print(f'  → Saved: ensemble_report.txt, ensemble.png')
    return {'acc_gcn': acc_gcn, 'acc_v17': acc_v17,
            'best_alpha': best_alpha, 'best_acc': best_acc}


# ────────────────── SUMMARY REPORT ───────────────────────────────

def write_summary(args, attn_stats, noise_results, error_stats,
                  ensemble_stats, output_dir, elapsed):
    sigmas   = args.sigmas
    baseline = noise_results[0.0]
    lines = [
        '=' * 65,
        '  FULL EVALUATION SUMMARY — NTU-60 V17',
        f'  OnTheFly  : {args.onthefly_weights}',
        f'  CTR-GCN   : {args.ctrgcn_weights}',
        f'  Split     : {args.split}',
        f'  Elapsed   : {elapsed:.1f}s',
        '=' * 65,
        '',
        '  [ACCURACY]',
        f'    CTR-GCN Only      : {error_stats["acc_gcn"]:.2f}%',
        f'    OnTheFly V17      : {error_stats["acc_v17"]:.2f}%',
        f'    Ensemble (best)   : {ensemble_stats["best_acc"]:.2f}%'
        f'  (alpha={ensemble_stats["best_alpha"]})',
        '',
        '  [ATTENTION QUALITY]',
        f'    Spatial Entropy   : {attn_stats["entropy"]:.3f}'
        f'  (uniform={attn_stats["uniform_entropy"]:.3f})'
        + ('  ✓' if attn_stats['entropy'] < attn_stats['uniform_entropy'] else '  ✗'),
        f'    Discrimination Gap: {attn_stats["disc_gap"]:.4f}'
        + ('  ✓' if attn_stats['disc_gap'] > 0 else '  ✗'),
        f'    Peak-to-Mean Ratio: {attn_stats["peak_mean"]:.2f}',
        '',
        '  [NOISE ROBUSTNESS]',
    ]
    for s in sigmas:
        delta = noise_results[s] - baseline
        ok = '✓' if delta > -5 or s <= 0.05 else ''
        lines.append(f'    σ={s:.3f} : {noise_results[s]:.2f}%  ({delta:+.2f}%) {ok}')
    lines += [
        '',
        '  [ERROR ANALYSIS vs CTR-GCN]',
        f'    V17 Improved      : +{error_stats["improved"]} samples',
        f'    V17 Blind Spots   : -{error_stats["blind_spots"]} samples',
        f'    Net Gain          : {error_stats["improved"]-error_stats["blind_spots"]:+d} samples',
        '',
        '  [OUTPUT FILES]',
        '    attention_report.txt   attention_perclass.png',
        '    attention_similarity.png   attention_metrics.png',
        '    noise_report.txt   noise_accuracy.png',
        '    error_analysis.txt   error_analysis.png   confidence_hist.png',
        '    ensemble_report.txt   ensemble.png',
        '    summary.txt',
        '=' * 65,
    ]
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n')
    print('\n'.join(lines))
    print(f'\nAll results saved to: {output_dir}')


# ─────────────────────────── MAIN ────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    print(f'\nDevice     : {DEVICE}')
    print(f'Output dir : {args.output_dir}')
    print(f'Split      : {args.split}')

    # Load models
    print('\n>>> Loading models...')
    ctrgcn_model   = load_ctrgcn(args.ctrgcn_weights, DEVICE)
    onthefly_model = load_onthefly(args.onthefly_weights, args.ctrgcn_weights, DEVICE)

    # Load dataset once
    print('\n>>> Loading validation dataset...')
    loader, dataset = make_loader(args)
    print(f'  Val samples: {len(dataset)}')

    # Base inference (collect all scores + attention in 1 pass)
    gcn_scores, v17_scores, labels, attn_data = run_base_inference(
        ctrgcn_model, onthefly_model, loader
    )

    # 1. Attention analysis
    attn_stats = analyze_attention(attn_data, labels, args.output_dir)

    # 2. Gaussian noise (re-use dataset, new loaders per sigma/trial)
    noise_results = run_gaussian_noise(onthefly_model, dataset, args, args.output_dir)

    # 3. Error analysis (CTR-GCN vs V17)
    error_stats = run_error_analysis(gcn_scores, v17_scores, labels, args.output_dir)

    # 4. Ensemble
    ensemble_stats = run_ensemble(gcn_scores, v17_scores, labels, args.output_dir)

    # Summary
    write_summary(args, attn_stats, noise_results, error_stats,
                  ensemble_stats, args.output_dir, time.time() - t0)


if __name__ == '__main__':
    main()
