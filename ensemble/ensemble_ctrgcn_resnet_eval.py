"""
Ensemble evaluation: CTR-GCN (skeleton) + ResNet50 (RGB weighted images)

Usage:
    python ensemble/ensemble_ctrgcn_resnet_eval.py
    python ensemble/ensemble_ctrgcn_resnet_eval.py --alpha 0.5
    python ensemble/ensemble_ctrgcn_resnet_eval.py --ctrgcn_weights ./result/nucla/CTROGC-GCN.pt --resnet_weights ./result/nucla/resnet_weight_group.pt
"""
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

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
    parser.add_argument('--rgb_path', type=str, default='../drive/MyDrive/Data/ucla_stroi_weighted/',
                        help='Path to weighted RGB images')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for CTR-GCN score. Final = ResNet + alpha * CTR-GCN')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    return parser.parse_args()


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
    
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  ✓ CTR-GCN loaded from: {weights_path}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params:,}")
    return model


def load_resnet(weights_path, device):
    """Load ResNet50 model with trained weights."""
    model = ResNet_Model(num_class=NUM_CLASS, pretrained=False).to(device)
    
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  ✓ ResNet50 loaded from: {weights_path}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params:,}")
    return model


def evaluate_ctrgcn(model, dataloader, device):
    """Evaluate CTR-GCN model, return scores and labels."""
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for data, _rgb, label, _idx in tqdm(dataloader, desc='CTR-GCN Inference'):
            data = data.float().to(device)
            output = model(data)
            all_scores.append(output.cpu().numpy())
            all_labels.append(label.numpy() if isinstance(label, torch.Tensor) else np.array(label))
    
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return scores, labels


def evaluate_resnet(model, dataloader, device):
    """Evaluate ResNet model, return scores and labels."""
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for rgb, label, _filename in tqdm(dataloader, desc='ResNet Inference'):
            rgb = rgb.float().to(device)
            output = model(rgb)
            all_scores.append(output.cpu().numpy())
            all_labels.append(label.numpy() if isinstance(label, torch.Tensor) else np.array([label]))
    
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return scores, labels


def compute_accuracy(scores, labels, label_names=None):
    """Compute overall and per-class accuracy."""
    preds = np.argmax(scores, axis=1)
    correct = (preds == labels).sum()
    total = len(labels)
    acc = correct / total
    
    # Per-class
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
    skeleton_loader = torch.utils.data.DataLoader(
        skeleton_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    print(f"  Skeleton val samples: {len(skeleton_dataset)}")
    
    # RGB data for ResNet
    resnet_dataset = ResNetFeeder(
        label_path='val',
        rgb_path=args.rgb_path,
    )
    resnet_loader = torch.utils.data.DataLoader(
        resnet_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    print(f"  RGB val samples: {len(resnet_dataset)}")
    
    # ---- Evaluate individual models ----
    print("\n>>> Evaluating CTR-GCN...")
    ctrgcn_scores, ctrgcn_labels = evaluate_ctrgcn(ctrgcn_model, skeleton_loader, DEVICE)
    
    print("\n>>> Evaluating ResNet...")
    resnet_scores, resnet_labels = evaluate_resnet(resnet_model, resnet_loader, DEVICE)
    
    # ---- Individual results ----
    acc_c, cor_c, tot_c, cls_c = compute_accuracy(ctrgcn_scores, ctrgcn_labels)
    print_results('CTR-GCN (Skeleton Only)', acc_c, cor_c, tot_c, cls_c)
    
    acc_r, cor_r, tot_r, cls_r = compute_accuracy(resnet_scores, resnet_labels)
    print_results('ResNet50 (RGB Weighted Only)', acc_r, cor_r, tot_r, cls_r)
    
    # ---- Ensemble fusion ----
    # Both datasets should have the same samples in the same order
    # Normalize scores (softmax) before fusion for fair comparison
    from scipy.special import softmax
    ctrgcn_norm = softmax(ctrgcn_scores, axis=1)
    resnet_norm = softmax(resnet_scores, axis=1)
    
    print(f"\n>>> Ensemble Fusion: ResNet + {args.alpha} * CTR-GCN")
    
    ensemble_scores = resnet_norm + args.alpha * ctrgcn_norm
    
    # Use ctrgcn_labels as ground truth (both should be the same)
    acc_e, cor_e, tot_e, cls_e = compute_accuracy(ensemble_scores, ctrgcn_labels)
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
        acc = (preds == ctrgcn_labels).sum() / len(ctrgcn_labels)
        marker = ' ★' if acc > best_acc or (acc == best_acc and alpha == best_alpha) else ''
        print(f'  {alpha:<10.1f} {acc*100:<12.2f}{marker}')
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
    
    print(f'\n  Best alpha: {best_alpha} → Accuracy: {best_acc*100:.2f}%')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
