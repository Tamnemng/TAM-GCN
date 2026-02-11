"""
Standalone training script cho ST-GCN 4-group model.
Không phụ thuộc vào processor framework phức tạp.

Usage:
    python tools/train_stgcn_group.py
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.getcwd())

from models.stgcn import Model
from feeder.feeder_nucla_group import Feeder, GROUP_NAMES

# ============ CONFIG ============
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_CLASS = 5
NUM_POINT = 20
NUM_PERSON = 1
IN_CHANNELS = 3
GRAPH = 'graph.ucla.Graph'
GRAPH_ARGS = {'labeling_mode': 'spatial'}

DATA_PATH = '../drive/MyDrive/Data/Data/NW-UCLA-ALL/'
BATCH_SIZE = 16
TEST_BATCH_SIZE = 64
NUM_EPOCHS = 80
BASE_LR = 0.1
LR_STEPS = [50, 65]
LR_DECAY = 0.1
WEIGHT_DECAY = 0.0001
WARM_UP_EPOCH = 5
REPEAT_TRAIN = 5

SAVE_DIR = './result/nucla/stgcn_group'
# ================================


def main():
    print("=" * 60)
    print("ST-GCN 4-Group Training")
    print("=" * 60)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Model
    print("\n>>> Khởi tạo model...")
    model = Model(
        in_channels=IN_CHANNELS,
        num_class=NUM_CLASS,
        num_point=NUM_POINT,
        num_person=NUM_PERSON,
        graph=GRAPH,
        graph_args=GRAPH_ARGS,
        edge_importance_weighting=True
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Data
    print("\n>>> Loading data...")
    train_dataset = Feeder(
        data_path=DATA_PATH,
        label_path='train',
        repeat=REPEAT_TRAIN,
        random_choose=True,
        random_shift=False,
        random_move=False,
        window_size=52,
        normalization=False
    )
    
    test_dataset = Feeder(
        data_path=DATA_PATH,
        label_path='val',
        repeat=1,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=52,
        normalization=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=2, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
        num_workers=2
    )
    
    # Loss & Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=BASE_LR,
        momentum=0.9,
        nesterov=True,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_STEPS, gamma=LR_DECAY)
    
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # === Warm up ===
        if epoch < WARM_UP_EPOCH:
            lr = BASE_LR * (epoch + 1) / WARM_UP_EPOCH
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # === Train ===
        model.train()
        train_loss = []
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
        for data, label, _ in pbar:
            data = data.float().to(DEVICE)
            label = label.long().to(DEVICE)
            
            output = model(data)
            loss = loss_fn(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            pred = output.argmax(dim=1)
            train_correct += (pred == label).sum().item()
            train_total += label.size(0)
            
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        train_acc = train_correct / train_total
        
        # === Test ===
        model.eval()
        test_correct = 0
        test_total = 0
        group_correct = {g: 0 for g in range(NUM_CLASS)}
        group_total = {g: 0 for g in range(NUM_CLASS)}
        
        with torch.no_grad():
            for data, label, _ in test_loader:
                data = data.float().to(DEVICE)
                label = label.long().to(DEVICE)
                
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += (pred == label).sum().item()
                test_total += label.size(0)
                
                for g in range(NUM_CLASS):
                    mask = label == g
                    group_correct[g] += (pred[mask] == label[mask]).sum().item()
                    group_total[g] += mask.sum().item()
        
        test_acc = test_correct / test_total
        
        if epoch >= WARM_UP_EPOCH:
            scheduler.step()
        
        # Print
        lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch+1}: loss={np.mean(train_loss):.4f}, '
              f'train_acc={train_acc*100:.1f}%, test_acc={test_acc*100:.1f}%, lr={lr:.5f}')
        for g in range(NUM_CLASS):
            g_acc = group_correct[g] / max(group_total[g], 1) * 100
            print(f'  Group {g} ({GROUP_NAMES[g]}): {g_acc:.1f}% ({group_correct[g]}/{group_total[g]})')
        
        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(SAVE_DIR, 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f'  ★ New best: {best_acc*100:.1f}% → saved to {save_path}')
    
    # === Extract Per-Group Importance (Gradient-based) ===
    print("\n" + "=" * 60)
    print("Extracting Class-Specific Importance (Gradient Analysis)")
    print("=" * 60)
    
    model.eval()
    
    # Define TARGET_JOINTS
    TARGET_JOINTS = {
        "head": [2, 3],           # neck, head
        "l_hand": [4, 5, 6, 7],   # l_shoulder → l_hand
        "r_hand": [8, 9, 10, 11], # r_shoulder → r_hand
        "l_leg": [12, 13, 14, 15], # l_hip → l_foot
        "r_leg": [16, 17, 18, 19], # r_hip → r_foot
    }
    
    # Store gradients per group
    # group_grads[g][part] = list of gradients
    group_grads = {g: {p: [] for p in TARGET_JOINTS.keys()} for g in range(NUM_CLASS)}
    
    # Enable gradients on input for analysis
    # We need to run forward pass on a subset of data
    analyze_subset_size = 200  # Analyze 200 samples per group (approx)
    
    print(f"analyzing gradients on training set ({analyze_subset_size} samples)...")
    
    analyzed_count = {g: 0 for g in range(NUM_CLASS)}
    
    for data, label, _ in tqdm(train_loader, desc="Analyzing Gradients"):
        if all(c >= analyze_subset_size for c in analyzed_count.values()):
            break
            
        data = data.float().to(DEVICE)
        data.requires_grad = True
        label = label.long().to(DEVICE)
        
        output = model(data)
        
        # Compute gradient for true class
        # score = output[torch.arange(B), label]
        score = torch.gather(output, 1, label.unsqueeze(1)).squeeze()
        score.sum().backward()
        
        # Get input gradient: (B, C, T, V, M)
        # We want importance per node V: sum over C, T, M
        grad = data.grad.abs().sum(dim=(1, 2, 4))  # (B, V)
        grad = grad.detach().cpu().numpy()
        
        labels_np = label.detach().cpu().numpy()
        
        for i in range(len(labels_np)):
            g = labels_np[i]
            if analyzed_count[g] < analyze_subset_size:
                # Map 20 joints to 5 body parts
                for part, joints in TARGET_JOINTS.items():
                    # Average gradient of joints in this part
                    part_grad = np.mean([grad[i, j] for j in joints])
                    group_grads[g][part].append(part_grad)
                analyzed_count[g] += 1
    
    # Compute average importance per group
    final_group_weights = {}
    
    print("\n>>> Per-Group Body Part Importance:")
    for g in range(NUM_CLASS):
        print(f"\nGroup {g}: {GROUP_NAMES[g]}")
        avg_grads = {}
        for part in TARGET_JOINTS.keys():
            if len(group_grads[g][part]) > 0:
                avg_grads[part] = np.mean(group_grads[g][part])
            else:
                avg_grads[part] = 0.0
        
        # Normalize to max 1.0 per group
        max_val = max(avg_grads.values()) if avg_grads else 1.0
        if max_val == 0: max_val = 1.0
        
        final_group_weights[g] = {}
        for part, val in avg_grads.items():
            norm_val = val / max_val
            final_group_weights[g][part] = float(norm_val)
            bar = '█' * int(norm_val * 20)
            print(f"  {part:>8s}: {norm_val:.4f} {bar}")

    # Save to JSON
    import json
    weights_path = os.path.join(SAVE_DIR, 'group_weights.json')
    with open(weights_path, 'w') as f:
        json.dump(final_group_weights, f, indent=2)
    print(f"\n>>> Saved group weights to: {weights_path}")
    print(">>> Dùng file này cho gen_ucla_stroi_weighted_stgcn.py")


if __name__ == '__main__':
    main()
