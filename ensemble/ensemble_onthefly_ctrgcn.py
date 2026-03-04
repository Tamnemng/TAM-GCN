"""
Ensemble evaluation: On-the-Fly ResNet (skeleton+RGB) + CTR-GCN (skeleton)

Loads both models, runs inference on the val set, and fuses softmax scores.
The on-the-fly ResNet model already uses CTR-GCN features internally for weighting,
but its final classification scores can still benefit from ensembling with
the standalone CTR-GCN's classification scores.

Usage:
    python ensemble/ensemble_onthefly_ctrgcn.py
    python ensemble/ensemble_onthefly_ctrgcn.py --alpha 1.0
    python ensemble/ensemble_onthefly_ctrgcn.py --onthefly_weights work_dir/nucla/resnet_onthefly/epoch61_model.pt
"""
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm

sys.path.append(os.getcwd())

from models.ctrgcn import Model as CTR_GCN_Model
from models.resnet_ctrgcn_ontheflyv2 import Model as OnTheFlyModel
from feeder.feeder_nucla_fused_ctr_resnet import Feeder as FusedFeeder

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_CLASS = 10
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


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble On-the-Fly ResNet + CTR-GCN')
    parser.add_argument('--ctrgcn_weights', type=str, default='./result/nucla/CTROGC-GCN.pt')
    parser.add_argument('--onthefly_weights', type=str, default='./work_dir/nucla/resnet_onthefly/epoch61_model.pt')
    parser.add_argument('--data_path', type=str, default='C:/Users/nguyn/Downloads/NW-UCLA-ALL/NW-UCLA-ALL')
    parser.add_argument('--rgb_path', type=str, default='C:/ucla_stroi/')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for CTR-GCN. Final = OnTheFly + alpha * CTR-GCN')
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


def load_weights_robust(weights_path, device):
    raw = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        if 'model_state_dict' in raw:
            raw = raw['model_state_dict']
        elif 'state_dict' in raw:
            raw = raw['state_dict']
    new_state = OrderedDict()
    for k, v in raw.items():
        name = k.replace('module.', '')
        new_state[name] = v
    return new_state


def load_ctrgcn(weights_path, device):
    model = CTR_GCN_Model(
        num_class=NUM_CLASS, num_point=20, num_person=1,
        graph='graph.ucla.Graph', graph_args={'labeling_mode': 'spatial'},
    ).to(device)
    state_dict = load_weights_robust(weights_path, device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  ✓ CTR-GCN loaded: {weights_path}")
    return model


def load_onthefly(weights_path, ctrgcn_weights_path, device):
    """Load the on-the-fly ResNet model and inject frozen CTR-GCN."""
    model = OnTheFlyModel(num_class=NUM_CLASS, pretrained=False).to(device)
    
    # Inject CTR-GCN for on-the-fly weighting
    ctrgcn_for_model = CTR_GCN_Model(
        num_class=NUM_CLASS, num_point=20, num_person=1,
        graph='graph.ucla.Graph', graph_args={'labeling_mode': 'spatial'},
    ).to(device)
    ctrgcn_state = load_weights_robust(ctrgcn_weights_path, device)
    ctrgcn_for_model.load_state_dict(ctrgcn_state, strict=False)
    ctrgcn_for_model.eval()
    for p in ctrgcn_for_model.parameters():
        p.requires_grad = False
    model.ctrgcn = ctrgcn_for_model
    
    # Load ResNet weights
    state_dict = load_weights_robust(weights_path, device)
    # Filter to only load resnet.* and buffer keys (skip ctrgcn.* since we loaded it above)
    model_state = model.state_dict()
    loaded = 0
    for k, v in state_dict.items():
        if k in model_state and model_state[k].size() == v.size():
            model_state[k] = v
            loaded += 1
    model.load_state_dict(model_state)
    model.eval()
    print(f"  ✓ On-the-Fly ResNet loaded ({loaded} params): {weights_path}")
    return model


def compute_accuracy(scores, labels):
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
    print('  ENSEMBLE: On-the-Fly ResNet + CTR-GCN')
    print('=' * 60)
    print(f'  OnTheFly weights: {args.onthefly_weights}')
    print(f'  CTR-GCN weights : {args.ctrgcn_weights}')
    print(f'  Alpha (CTR-GCN) : {args.alpha}')
    print(f'  Device          : {DEVICE}')
    print()
    
    # ---- Load models ----
    print(">>> Loading models...")
    ctrgcn_model = load_ctrgcn(args.ctrgcn_weights, DEVICE)
    onthefly_model = load_onthefly(args.onthefly_weights, args.ctrgcn_weights, DEVICE)
    
    # ---- Load val data ----
    # The fused feeder returns (skeleton, rgb, label)
    # We use the same feeder for both models
    print("\n>>> Loading validation data...")
    val_dataset = FusedFeeder(
        data_path=args.data_path,
        label_path='val',
        rgb_path=args.rgb_path,
        temporal_rgb_frames=5,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=52,
        normalization=False,
    )
    print(f"  Val samples: {len(val_dataset)}")
    
    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    # ---- Run inference ----
    print("\n>>> Running inference...")
    ctrgcn_all_scores = []
    onthefly_all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for skeleton, rgb, label in tqdm(loader, desc='Inference'):
            skeleton = skeleton.float().to(DEVICE)
            rgb = rgb.float().to(DEVICE)
            
            # CTR-GCN: only uses skeleton
            ctrgcn_out = ctrgcn_model(skeleton)
            ctrgcn_all_scores.append(ctrgcn_out.cpu().numpy())
            
            # On-the-fly ResNet: uses skeleton + rgb
            onthefly_out = onthefly_model(skeleton, rgb)
            onthefly_all_scores.append(onthefly_out.cpu().numpy())
            
            if isinstance(label, torch.Tensor):
                all_labels.append(label.numpy())
            else:
                all_labels.append(np.array(label))
    
    ctrgcn_scores = np.concatenate(ctrgcn_all_scores)
    onthefly_scores = np.concatenate(onthefly_all_scores)
    labels = np.concatenate(all_labels)
    
    # ---- Individual results ----
    acc_c, cor_c, tot_c, cls_c = compute_accuracy(ctrgcn_scores, labels)
    print_results('CTR-GCN (Skeleton Only)', acc_c, cor_c, tot_c, cls_c)
    
    acc_o, cor_o, tot_o, cls_o = compute_accuracy(onthefly_scores, labels)
    print_results('On-the-Fly ResNet (Skeleton+RGB)', acc_o, cor_o, tot_o, cls_o)
    
    # ---- Ensemble fusion ----
    from scipy.special import softmax
    ctrgcn_norm = softmax(ctrgcn_scores, axis=1)
    onthefly_norm = softmax(onthefly_scores, axis=1)
    
    ensemble_scores = onthefly_norm + args.alpha * ctrgcn_norm
    acc_e, cor_e, tot_e, cls_e = compute_accuracy(ensemble_scores, labels)
    print_results(f'ENSEMBLE (OnTheFly + {args.alpha} * CTR-GCN)', acc_e, cor_e, tot_e, cls_e)
    
    # ---- Try multiple alpha values ----
    print(f'\n>>> Trying different alpha values...')
    print(f'  {"Alpha":<10s} {"Accuracy":<12s}')
    print(f'  {"-"*22}')
    best_alpha = args.alpha
    best_acc = acc_e
    
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        combo = onthefly_norm + alpha * ctrgcn_norm
        preds = np.argmax(combo, axis=1)
        acc = (preds == labels).sum() / len(labels)
        marker = ' ★' if alpha == best_alpha else ''
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            marker = ' ★'
        print(f'  {alpha:<10.1f} {acc*100:<12.2f}{marker}')
    
    print(f'\n  Best alpha: {best_alpha} → Accuracy: {best_acc*100:.2f}%')
    
    # ---- Summary ----
    print(f'\n{"="*60}')
    print(f'  SUMMARY')
    print(f'{"="*60}')
    print(f'  CTR-GCN Only:       {acc_c*100:.2f}%')
    print(f'  OnTheFly ResNet:    {acc_o*100:.2f}%')
    print(f'  Ensemble (α={best_alpha}):  {best_acc*100:.2f}%')
    improvement = (best_acc - max(acc_c, acc_o)) * 100
    print(f'  Improvement:        {improvement:+.2f}% over best single model')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
