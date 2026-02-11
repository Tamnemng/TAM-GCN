"""
Script phân tích trọng số body part theo từng label (action class).
Kiểm tra xem trong cùng 1 action, các sample có weight giống nhau không.
Nếu weight quá khác nhau (std cao) → CTR-GCN đang cho importance không nhất quán
→ weight sẽ gây nhiễu cho ResNet thay vì giúp ích.
"""
import torch
import numpy as np
import sys
import os
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.getcwd())
from models.ctrgcn import Model as CTRGCN
from feeder.feeder_nucla_fusion import Feeder

output_device = 0 if torch.cuda.is_available() else 'cpu'
device = torch.device(f"cuda:{output_device}" if torch.cuda.is_available() else "cpu")

NUM_BODY_PARTS = 5
TARGET_JOINTS = [3, 11, 7, 18, 14]
PARTS_NAMES = ['head', 'l_hand', 'r_hand', 'l_leg', 'r_leg']

# UCLA action labels (0-indexed, label trong data_dict là 1-indexed)
ACTION_NAMES = {
    1: 'pick up with one hand',
    2: 'pick up with two hands', 
    3: 'drop trash',
    4: 'walk around',
    5: 'sit down',
    6: 'stand up',
    7: 'donning',
    8: 'doffing',
    9: 'throw',
    10: 'carry',
}


def analyze_weights(weights_path):
    print(">>> Đang khởi tạo mô hình CTR-GCN...")
    graph_args = {'labeling_mode': 'spatial'}
    model_ske = CTRGCN(
        num_class=10, 
        num_point=20, 
        num_person=1,
        graph='graph.ucla.Graph',
        graph_args=graph_args,
        in_channels=3
    ).to(device)

    if weights_path and os.path.exists(weights_path):
        print(f">>> Loading weights từ: {weights_path}")
        try:
            model_ske.load_state_dict(torch.load(weights_path))
            print(">>> Load weights thành công!")
        except Exception as e:
            print(f"!!! Không load được weights ({e}). Dùng weights ngẫu nhiên.")
    else:
        print("!!! Không tìm thấy weights. Dùng weights ngẫu nhiên → kết quả sẽ vô nghĩa!")

    model_ske.eval()

    # Thu thập weights theo label
    # Key: label, Value: list of weight arrays (mỗi array có 5 phần tử)
    weights_by_label = defaultdict(list)
    
    for split in ['train', 'val']:
        print(f"\n>>> Đang xử lý: {split}")
        feeder = Feeder(
            split=split,
            random_choose=False,
            random_shift=False,
            random_move=False,
            window_size=50,
            temporal_rgb_frames=5
        )
        
        loader = torch.utils.data.DataLoader(
            dataset=feeder, batch_size=1, shuffle=False, num_workers=0
        )

        for i, (data, label, index) in enumerate(tqdm(loader)):
            data_ske = data[0].float().to(device)
            lbl = label.item()
            
            N, C, T, V, M = data_ske.size()
            
            with torch.no_grad():
                _, feature = model_ske.extract_feature(data_ske)
                
            intensity_s = (feature * feature).sum(dim=1) ** 0.5
            intensity_s = intensity_s.cpu().detach().numpy()
            feature_s = np.abs(intensity_s)
            
            feat_min, feat_max = feature_s.min(), feature_s.max()
            if (feat_max - feat_min) > 0:
                feature_s = (feature_s - feat_min) / (feat_max - feat_min)
            
            weights_per_part = np.ones(NUM_BODY_PARTS)
            n = 0
            person_idx = 0
            
            _, _, V_feat, M_feat = feature_s.shape
            
            temporal_positions = 15
            for j, v_idx in enumerate(TARGET_JOINTS):
                if v_idx < V_feat:
                    feature_val = feature_s[n, :, v_idx, person_idx]
                    k = min(temporal_positions, len(feature_val))
                    top_k_vals = np.partition(feature_val, -k)[-k:]
                    weights_per_part[j] = top_k_vals.mean()
            
            # Normalize về [0.5, 1.5] như script chính
            w_min, w_max = weights_per_part.min(), weights_per_part.max()
            if (w_max - w_min) > 0:
                weights_per_part = 0.5 + 1.0 * (weights_per_part - w_min) / (w_max - w_min)
            else:
                weights_per_part = np.ones(NUM_BODY_PARTS)
            
            weights_by_label[lbl].append(weights_per_part)

    # ===== IN KẾT QUẢ =====
    print("\n" + "=" * 80)
    print("PHÂN TÍCH TRỌNG SỐ BODY PART THEO TỪNG LABEL")
    print("=" * 80)
    print(f"{'':>5} | {'head':>12} | {'l_hand':>12} | {'r_hand':>12} | {'l_leg':>12} | {'r_leg':>12} | {'Consistency':>12}")
    print("-" * 95)
    
    for lbl in sorted(weights_by_label.keys()):
        ws = np.array(weights_by_label[lbl])  # (num_samples, 5)
        mean_w = ws.mean(axis=0)
        std_w = ws.std(axis=0)
        avg_std = std_w.mean()
        
        action = ACTION_NAMES.get(lbl, f'action_{lbl}')
        
        # In trung bình
        w_str = " | ".join([f"{mean_w[j]:.3f}±{std_w[j]:.2f}" for j in range(NUM_BODY_PARTS)])
        
        # Đánh giá consistency: std thấp = nhất quán, std cao = không nhất quán
        if avg_std < 0.05:
            consistency = "✓ Tốt"
        elif avg_std < 0.12:
            consistency = "~ Trung bình"
        else:
            consistency = "✗ Kém"
        
        print(f"L{lbl:>2}  | {w_str} | {consistency:>12}")
        
    print("-" * 95)
    
    # In phân tích chi tiết
    print("\n" + "=" * 80)
    print("CHI TIẾT: Body part nào bị LÀM MỜ / TĂNG SÁNG theo từng action")
    print("=" * 80)
    
    for lbl in sorted(weights_by_label.keys()):
        ws = np.array(weights_by_label[lbl])
        mean_w = ws.mean(axis=0)
        std_w = ws.std(axis=0)
        avg_std = std_w.mean()
        
        action = ACTION_NAMES.get(lbl, f'action_{lbl}')
        n_samples = len(ws)
        
        print(f"\n--- Label {lbl}: {action} ({n_samples} samples, avg_std={avg_std:.4f}) ---")
        
        # Sắp xếp body parts theo importance
        sorted_parts = np.argsort(mean_w)  # ascending → phần tử đầu = ít quan trọng nhất
        
        darkened = []
        brightened = []
        neutral = []
        
        for idx in sorted_parts:
            w = mean_w[idx]
            s = std_w[idx]
            name = PARTS_NAMES[idx]
            if w < 0.8:
                darkened.append(f"{name} (w={w:.3f}±{s:.2f})")
            elif w > 1.2:
                brightened.append(f"{name} (w={w:.3f}±{s:.2f})")
            else:
                neutral.append(f"{name} (w={w:.3f}±{s:.2f})")
        
        if brightened:
            print(f"  🔆 Tăng sáng: {', '.join(brightened)}")
        if neutral:
            print(f"  ⚪ Trung tính: {', '.join(neutral)}")
        if darkened:
            print(f"  🔅 Làm mờ:    {', '.join(darkened)}")
        
        if avg_std > 0.12:
            print(f"  ⚠️  STD CAO ({avg_std:.3f}): Weight KHÔNG nhất quán giữa các sample → gây nhiễu!")


if __name__ == '__main__':
    WEIGHTS_PATH = './result/nucla/CTROGC-GCN.pt'
    analyze_weights(WEIGHTS_PATH)
