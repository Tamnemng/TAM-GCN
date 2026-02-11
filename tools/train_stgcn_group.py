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
NUM_CLASS = 4
NUM_POINT = 20
NUM_PERSON = 1
IN_CHANNELS = 3
GRAPH = 'graph.ucla.Graph'
GRAPH_ARGS = {'labeling_mode': 'spatial'}

DATA_PATH = 'joint'
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
    
    # === Done — Print edge importance ===
    print("\n" + "=" * 60)
    print(f"Training hoàn tất! Best accuracy: {best_acc*100:.1f}%")
    print("=" * 60)
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pt')))
    
    # Print edge importance per joint
    joint_importance = model.get_edge_importance_per_joint()
    
    UCLA_JOINT_NAMES = [
        'hip_center', 'spine', 'neck', 'head',          # 0-3
        'l_shoulder', 'l_elbow', 'l_wrist', 'l_hand',   # 4-7
        'r_shoulder', 'r_elbow', 'r_wrist', 'r_hand',   # 8-11
        'l_hip', 'l_knee', 'l_ankle', 'l_foot',         # 12-15
        'r_hip', 'r_knee', 'r_ankle', 'r_foot',         # 16-19
    ]
    
    print("\n>>> Edge Importance per Joint (normalized):")
    sorted_idx = np.argsort(joint_importance)[::-1]
    for idx in sorted_idx:
        bar = '█' * int(joint_importance[idx] * 30)
        print(f"  Joint {idx:2d} ({UCLA_JOINT_NAMES[idx]:>12s}): {joint_importance[idx]:.4f} {bar}")
    
    # Map to 5 body parts
    TARGET_JOINTS = {
        'head': [2, 3],           # neck, head
        'l_hand': [4, 5, 6, 7],   # l_shoulder → l_hand
        'r_hand': [8, 9, 10, 11], # r_shoulder → r_hand
        'l_leg': [12, 13, 14, 15], # l_hip → l_foot
        'r_leg': [16, 17, 18, 19], # r_hip → r_foot
    }
    
    print("\n>>> Body Part Importance (for FiveFS weighting):")
    body_part_weights = {}
    for part, joints in TARGET_JOINTS.items():
        w = np.mean([joint_importance[j] for j in joints])
        body_part_weights[part] = w
        bar = '█' * int(w * 30)
        print(f"  {part:>8s}: {w:.4f} {bar}")
    
    # Save body part weights
    import json
    weights_path = os.path.join(SAVE_DIR, 'body_part_weights.json')
    with open(weights_path, 'w') as f:
        json.dump(body_part_weights, f, indent=2)
    print(f"\n>>> Body part weights saved to: {weights_path}")
    print(">>> Dùng weights này trong gen_ucla_stroi_weighted.py để tạo ảnh weighted nhất quán!")


if __name__ == '__main__':
    main()
