"""
Ensemble evaluation: CTR-GCN (skeleton) + ResNet50 (RGB weighted images)

Fusion theo filename để đảm bảo đúng sample giữa 2 model.

Usage:
    python ensemble/ensemble_ctrgcn_resnet_eval.py
    python ensemble/ensemble_ctrgcn_resnet_eval.py --alpha 0.5
    python ensemble/ensemble_ctrgcn_resnet_eval.py --resnet_weights work_dir/nucla_123/resnet_only/best_model.pt
"""
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.getcwd())

from models.ctrgcn import Model as CTR_GCN_Model
from models.resnet_only import Model as ResNet_Model
from feeder.feeder_nucla_gcn import Feeder as SkeletonFeeder
from feeder.feeder_nucla_resnet import Feeder as ResNetFeeder

# ============ CONFIG ============
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

LABEL_NAMES = {
    0: 'Pick up with one hand',
    1: 'Pick up with two hands',
    2: 'Drop trash',
    3: 'Walk around',
    4: 'Sit down',
    5: 'Stand up',
    6: 'Donning',
    7: 'Doffing',
    8: 'Throw',
    9: 'Carry',
}
NUM_CLASS = 10
# ================================


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble CTR-GCN + ResNet50')
    parser.add_argument('--ctrgcn_weights', type=str, default='./result/nucla/CTROGC-GCN.pt',
                        help='Path to CTR-GCN model weights (.pt)')
    parser.add_argument('--resnet_weights', type=str, default='./result/nucla/resnet_weight_group.pt',
                        help='Path to ResNet model weights (.pt)')
    parser.add_argument('--data_path', type=str, default='../drive/MyDrive/Data/Data/NW-UCLA-ALL/',
                        help='Path to skeleton data root')
    parser.add_argument('--rgb_path', type=str, default='../drive/MyDrive/Data/ucla_stroi_weighted_stgcn/',
                        help='Path to weighted RGB images')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for CTR-GCN score. Final = ResNet + alpha * CTR-GCN')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    return parser.parse_args()


def load_weights_robust(weights_path, device):
    """Load weights với xử lý các format khác nhau (module. prefix, nested dict, etc.)"""
    raw = torch.load(weights_path, map_location=device)
    
    # Xử lý nested dict format
    if isinstance(raw, dict):
        if 'model_state_dict' in raw:
            raw = raw['model_state_dict']
        elif 'state_dict' in raw:
            raw = raw['state_dict']
    
    # Xử lý module. prefix (DataParallel)
    new_state = OrderedDict()
    for k, v in raw.items():
        name = k.replace('module.', '')
        new_state[name] = v
    
    return new_state


def load_ctrgcn(weights_path, device):
    """Load CTR-GCN model with trained weights."""
    model = CTR_GCN_Model(
        num_class=NUM_CLASS,
        num_point=20,
        num_person=1,
        graph='graph.ucla.Graph',
        graph_args={'labeling_mode': 'spatial'},
        in_channels=3,
    ).to(device)
    
    state_dict = load_weights_robust(weights_path, device)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"  ✓ CTR-GCN loaded (strict) from: {weights_path}")
    except RuntimeError as e:
        print(f"  ⚠ Strict load failed: {e}")
        print(f"  → Trying partial load...")
        model_state = model.state_dict()
        loaded = 0
        for k, v in state_dict.items():
            if k in model_state and model_state[k].size() == v.size():
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        print(f"  ✓ CTR-GCN partial loaded ({loaded}/{len(model_state)} params)")
    
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params:,}")
    return model


def load_resnet(weights_path, device):
    """Load ResNet50 model with trained weights."""
    model = ResNet_Model(num_class=NUM_CLASS, pretrained=False).to(device)
    
    state_dict = load_weights_robust(weights_path, device)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"  ✓ ResNet50 loaded (strict) from: {weights_path}")
    except RuntimeError as e:
        print(f"  ⚠ Strict load failed: {e}")
        print(f"  → Trying partial load...")
        model_state = model.state_dict()
        loaded = 0
        for k, v in state_dict.items():
            if k in model_state and model_state[k].size() == v.size():
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        print(f"  ✓ ResNet50 partial loaded ({loaded}/{len(model_state)} params)")
    
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params:,}")
    return model


def evaluate_ctrgcn_by_filename(model, dataset, batch_size, device):
    """
    Evaluate CTR-GCN model, return dict {filename: score_array}.
    feeder_nucla_gcn trả về (skeleton, rgb, label, index).
    data_dict[index]['file_name'] để lấy filename.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    scores_dict = {}  # filename -> score
    labels_dict = {}  # filename -> label
    
    sample_idx = 0
    with torch.no_grad():
        for data, _rgb, label, indices in tqdm(loader, desc='CTR-GCN Inference'):
            data = data.float().to(device)
            output = model(data)
            output_np = output.cpu().numpy()
            
            if isinstance(label, torch.Tensor):
                label_np = label.numpy()
            else:
                label_np = np.array(label)
            
            if isinstance(indices, torch.Tensor):
                indices_np = indices.numpy()
            else:
                indices_np = np.array(indices)
            
            for i in range(len(label_np)):
                idx = indices_np[i] % len(dataset.data_dict)
                fname = dataset.data_dict[idx]['file_name']
                scores_dict[fname] = output_np[i]
                labels_dict[fname] = label_np[i]
    
    return scores_dict, labels_dict


def evaluate_resnet_by_filename(model, dataset, batch_size, device):
    """
    Evaluate ResNet model, return dict {filename: score_array}.
    feeder_nucla_resnet trả về (rgb, label, filename).
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    scores_dict = {}  # filename -> score
    labels_dict = {}  # filename -> label
    
    with torch.no_grad():
        for rgb, label, filenames in tqdm(loader, desc='ResNet Inference'):
            rgb = rgb.float().to(device)
            output = model(rgb)
            output_np = output.cpu().numpy()
            
            if isinstance(label, torch.Tensor):
                label_np = label.numpy()
            else:
                label_np = np.array(label)
            
            for i in range(len(label_np)):
                fname = filenames[i] if isinstance(filenames, (list, tuple)) else filenames
                scores_dict[fname] = output_np[i]
                labels_dict[fname] = label_np[i]
    
    return scores_dict, labels_dict


def compute_accuracy(scores, labels):
    """Compute overall and per-class accuracy."""
    preds = np.argmax(scores, axis=1)
    correct = (preds == labels).sum()
    total = len(labels)
    acc = correct / total
    
    class_acc = {}
    for c in range(NUM_CLASS):
        mask = labels == c
        if mask.sum() > 0:
            c_correct = (preds[mask] == labels[mask]).sum()
            c_total = mask.sum()
            class_acc[c] = (c_correct, c_total, c_correct / c_total)
        else:
            class_acc[c] = (0, 0, 0.0)
    
    return acc, correct, total, class_acc


def print_results(title, acc, correct, total, class_acc):
    """Print formatted results."""
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}')
    print(f'  Top-1 Accuracy: {acc:.4f} ({acc*100:.2f}%)')
    print(f'  Correct: {correct}/{total}')
    print(f'{"-"*60}')
    
    for c in range(NUM_CLASS):
        c_correct, c_total, c_acc = class_acc[c]
        name = LABEL_NAMES.get(c, f'Class {c}')
        bar = '█' * int(c_acc * 20)
        print(f'  {c:2d}. {name:<25s}: {c_acc*100:5.1f}% ({c_correct}/{c_total}) {bar}')
    print(f'{"="*60}')


def plot_confusion_matrix(scores, labels, title, output_path):
    """Vẽ confusion matrix và lưu ra file PNG."""
    preds = np.argmax(scores, axis=1)
    acc = (preds == labels).sum() / len(labels)
    
    class_names = [LABEL_NAMES[i] for i in range(NUM_CLASS)]
    # Rút gọn tên cho đẹp trên heatmap
    short_names = [
        'Pick 1H', 'Pick 2H', 'Drop', 'Walk',
        'Sit', 'Stand', 'Donning', 'Doffing',
        'Throw', 'Carry'
    ]
    
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASS)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # --- Confusion matrix (counts) ---
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names,
                ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    axes[0].set_title(f'{title}\nConfusion Matrix (Counts) — Acc: {acc*100:.2f}%', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # --- Confusion matrix (normalized) ---
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=short_names, yticklabels=short_names,
                ax=axes[1], vmin=0, vmax=1, cbar_kws={'shrink': 0.8})
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    axes[1].set_title(f'{title}\nConfusion Matrix (Normalized) — Acc: {acc*100:.2f}%', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  📊 Confusion matrix saved: {output_path}')


def main():
    args = parse_args()
    
    print('=' * 60)
    print('  ENSEMBLE: CTR-GCN (Skeleton) + ResNet50 (RGB Weighted)')
    print('=' * 60)
    print(f'  CTR-GCN weights : {args.ctrgcn_weights}')
    print(f'  ResNet weights  : {args.resnet_weights}')
    print(f'  Data path       : {args.data_path}')
    print(f'  RGB path        : {args.rgb_path}')
    print(f'  Alpha (CTR-GCN) : {args.alpha}')
    print(f'  Device          : {DEVICE}')
    print()
    
    # ---- Load models ----
    print(">>> Loading models...")
    ctrgcn_model = load_ctrgcn(args.ctrgcn_weights, DEVICE)
    resnet_model = load_resnet(args.resnet_weights, DEVICE)
    
    # ---- Load data ----
    print("\n>>> Loading validation data...")
    
    # Skeleton data for CTR-GCN
    skeleton_dataset = SkeletonFeeder(
        data_path=args.data_path,
        label_path='val',
        repeat=1,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=52,
        normalization=False,
    )
    print(f"  Skeleton val samples: {len(skeleton_dataset)}")
    
    # RGB data for ResNet
    resnet_dataset = ResNetFeeder(
        label_path='val',
        rgb_path=args.rgb_path,
    )
    print(f"  RGB val samples: {len(resnet_dataset)}")
    
    # ---- Evaluate individual models (by filename) ----
    print("\n>>> Evaluating CTR-GCN...")
    ctrgcn_scores_dict, ctrgcn_labels_dict = evaluate_ctrgcn_by_filename(
        ctrgcn_model, skeleton_dataset, args.batch_size, DEVICE
    )
    
    print("\n>>> Evaluating ResNet...")
    resnet_scores_dict, resnet_labels_dict = evaluate_resnet_by_filename(
        resnet_model, resnet_dataset, args.batch_size, DEVICE
    )
    
    # ---- Report độc lập từng model ----
    # CTR-GCN
    ctrgcn_fnames = sorted(ctrgcn_scores_dict.keys())
    ctrgcn_scores_arr = np.array([ctrgcn_scores_dict[f] for f in ctrgcn_fnames])
    ctrgcn_labels_arr = np.array([ctrgcn_labels_dict[f] for f in ctrgcn_fnames])
    acc_c, cor_c, tot_c, cls_c = compute_accuracy(ctrgcn_scores_arr, ctrgcn_labels_arr)
    print_results('CTR-GCN (Skeleton Only)', acc_c, cor_c, tot_c, cls_c)
    
    # ResNet
    resnet_fnames = sorted(resnet_scores_dict.keys())
    resnet_scores_arr = np.array([resnet_scores_dict[f] for f in resnet_fnames])
    resnet_labels_arr = np.array([resnet_labels_dict[f] for f in resnet_fnames])
    acc_r, cor_r, tot_r, cls_r = compute_accuracy(resnet_scores_arr, resnet_labels_arr)
    print_results('ResNet50 (RGB Weighted Only)', acc_r, cor_r, tot_r, cls_r)
    
    # ---- Ensemble fusion (filename-matched) ----
    # Tìm common filenames giữa 2 model
    common_fnames = sorted(set(ctrgcn_scores_dict.keys()) & set(resnet_scores_dict.keys()))
    print(f"\n  Common samples for fusion: {len(common_fnames)}")
    
    if len(common_fnames) == 0:
        print("  ❌ Không có sample chung! Kiểm tra lại data.")
        print(f"     CTR-GCN filenames (mẫu): {ctrgcn_fnames[:5]}")
        print(f"     ResNet filenames (mẫu): {resnet_fnames[:5]}")
        return
    
    only_ctrgcn = set(ctrgcn_scores_dict.keys()) - set(resnet_scores_dict.keys())
    only_resnet = set(resnet_scores_dict.keys()) - set(ctrgcn_scores_dict.keys())
    if only_ctrgcn:
        print(f"  ⚠ {len(only_ctrgcn)} samples chỉ có ở CTR-GCN (bỏ qua)")
    if only_resnet:
        print(f"  ⚠ {len(only_resnet)} samples chỉ có ở ResNet (bỏ qua)")
    
    # Build aligned arrays
    ctrgcn_common = np.array([ctrgcn_scores_dict[f] for f in common_fnames])
    resnet_common = np.array([resnet_scores_dict[f] for f in common_fnames])
    labels_common = np.array([ctrgcn_labels_dict[f] for f in common_fnames])
    
    # Verify labels match
    resnet_labels_common = np.array([resnet_labels_dict[f] for f in common_fnames])
    if not np.array_equal(labels_common, resnet_labels_common):
        print("  ⚠ Labels không khớp giữa 2 feeder! Kiểm tra label mapping.")
        mismatch = (labels_common != resnet_labels_common).sum()
        print(f"    Số sample labels khác nhau: {mismatch}/{len(common_fnames)}")
        print(f"    CTR-GCN labels unique: {np.unique(labels_common)}")
        print(f"    ResNet labels unique:  {np.unique(resnet_labels_common)}")
    
    # Normalize scores (softmax) before fusion
    from scipy.special import softmax
    ctrgcn_norm = softmax(ctrgcn_common, axis=1)
    resnet_norm = softmax(resnet_common, axis=1)
    
    print(f"\n>>> Ensemble Fusion: ResNet + {args.alpha} * CTR-GCN")
    
    ensemble_scores = resnet_norm + args.alpha * ctrgcn_norm
    
    acc_e, cor_e, tot_e, cls_e = compute_accuracy(ensemble_scores, labels_common)
    print_results(f'ENSEMBLE (ResNet + {args.alpha} * CTR-GCN)', acc_e, cor_e, tot_e, cls_e)
    
    # ---- Summary comparison ----
    print(f'\n{"="*60}')
    print(f'  SUMMARY COMPARISON')
    print(f'{"="*60}')
    print(f'  CTR-GCN Only:    {acc_c*100:.2f}%')
    print(f'  ResNet Only:     {acc_r*100:.2f}%')
    print(f'  Ensemble:        {acc_e*100:.2f}%')
    improvement = (acc_e - max(acc_c, acc_r)) * 100
    print(f'  Improvement:     {improvement:+.2f}% over best single model')
    print(f'{"="*60}')
    
    # ---- Try multiple alpha values ----
    print(f'\n>>> Trying different alpha values...')
    print(f'  {"Alpha":<10s} {"Accuracy":<12s}')
    print(f'  {"-"*22}')
    best_alpha = args.alpha
    best_acc = acc_e
    
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        combo = resnet_norm + alpha * ctrgcn_norm
        preds = np.argmax(combo, axis=1)
        acc = (preds == labels_common).sum() / len(labels_common)
        marker = ' ★' if acc > best_acc or (acc == best_acc and alpha == best_alpha) else ''
        print(f'  {alpha:<10.1f} {acc*100:<12.2f}{marker}')
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
    
    print(f'\n  Best alpha: {best_alpha} → Accuracy: {best_acc*100:.2f}%')
    print(f'{"="*60}')
    
    # ---- Vẽ Confusion Matrix ----
    print(f'\n>>> Vẽ Confusion Matrix...')
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Confusion matrix cho alpha hiện tại
    plot_confusion_matrix(
        ensemble_scores, labels_common,
        f'Ensemble (ResNet + {args.alpha} × CTR-GCN)',
        os.path.join(save_dir, f'confusion_matrix_alpha_{args.alpha}.png')
    )
    
    # 2. Confusion matrix cho best alpha
    if best_alpha != args.alpha:
        best_scores = resnet_norm + best_alpha * ctrgcn_norm
        plot_confusion_matrix(
            best_scores, labels_common,
            f'Ensemble (ResNet + {best_alpha} × CTR-GCN) — BEST',
            os.path.join(save_dir, f'confusion_matrix_alpha_{best_alpha}_best.png')
        )
    
    # 3. Confusion matrix cho từng model riêng lẻ
    plot_confusion_matrix(
        ctrgcn_scores_arr, ctrgcn_labels_arr,
        'CTR-GCN (Skeleton Only)',
        os.path.join(save_dir, 'confusion_matrix_ctrgcn.png')
    )
    plot_confusion_matrix(
        resnet_scores_arr, resnet_labels_arr,
        'ResNet50 (RGB Weighted Only)',
        os.path.join(save_dir, 'confusion_matrix_resnet.png')
    )
    
    print(f'\n>>> Done! Tất cả confusion matrix đã lưu trong: {save_dir}')


if __name__ == '__main__':
    main()
