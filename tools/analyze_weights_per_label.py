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
import matplotlib.pyplot as plt
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

    # Thu thập 1 sample cho mỗi label
    collected_labels = set()
    output_dir = './weight_heatmaps'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\n>>> Sẽ lưu các biểu đồ nhiệt (heatmap) tại: {output_dir}")
    
    for split in ['train', 'val']:
        if len(collected_labels) >= 10:
            break
            
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
            lbl = label.item()
            
            # Chỉ lấy 1 sample cho mỗi label
            if lbl in collected_labels:
                continue
                
            data_ske = data[0].float().to(device)
            N, C, T, V, M = data_ske.size()
            
            with torch.no_grad():
                _, feature = model_ske.extract_feature(data_ske)
                
            intensity_s = (feature * feature).sum(dim=1) ** 0.5
            intensity_s = intensity_s.cpu().detach().numpy()
            feature_s = np.abs(intensity_s)
            
            feat_min, feat_max = feature_s.min(), feature_s.max()
            if (feat_max - feat_min) > 0:
                feature_s = (feature_s - feat_min) / (feat_max - feat_min)
            
            # Cấu trúc: 5 body parts, 5 frames
            num_frames = 5
            weights_per_part = np.ones((NUM_BODY_PARTS, num_frames))
            
            n = 0
            person_idx = 0
            
            _, T_feat, V_feat, M_feat = feature_s.shape
            
            # Chia T_feat thành 5 phần bằng nhau cho 5 frames
            segment_size = max(1, T_feat // num_frames)
            
            for f in range(num_frames):
                start_t = f * segment_size
                end_t = (f + 1) * segment_size if f < num_frames - 1 else T_feat
                
                temporal_positions = max(1, (end_t - start_t) // 2)
                
                for j, v_idx in enumerate(TARGET_JOINTS):
                    if v_idx < V_feat:
                        feature_val = feature_s[n, start_t:end_t, v_idx, person_idx]
                        k = min(temporal_positions, len(feature_val))
                        if k > 0:
                            top_k_vals = np.partition(feature_val, -k)[-k:]
                            weights_per_part[j, f] = top_k_vals.mean()
                        else:
                            weights_per_part[j, f] = 0.0
            
            # Normalize về [0.5, 1.5]
            w_min, w_max = weights_per_part.min(), weights_per_part.max()
            if (w_max - w_min) > 0:
                weights_per_part = 0.5 + 1.0 * (weights_per_part - w_min) / (w_max - w_min)
            else:
                weights_per_part = np.ones((NUM_BODY_PARTS, num_frames))
            
            # ----- Vẽ Heatmap và Lưu Ảnh -----
            plt.figure(figsize=(8, 6))
            
            # Trục y là Body Parts, Trục x là Frames
            # cmap 'Reds' hoặc 'viridis' để thể hiện heatmap
            plt.imshow(weights_per_part, cmap='viridis', aspect='auto')
            plt.colorbar(label='Trọng số (Weight)')
            
            plt.yticks(ticks=np.arange(NUM_BODY_PARTS), labels=PARTS_NAMES)
            plt.xticks(ticks=np.arange(num_frames), labels=[f"Frame {idx+1}" for idx in range(num_frames)])
            
            action_name = ACTION_NAMES.get(lbl, f'action_{lbl}')
            plt.title(f'Bản đồ nhiệt trọng số - Label {lbl}: {action_name}')
            
            plt.tight_layout()
            
            # Đánh index mảng
            filename = f"label_{lbl:02d}_{action_name.replace(' ', '_')}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=150)
            plt.close()
            
            collected_labels.add(lbl)
            
            # Dừng nếu đã đủ 10 labels
            if len(collected_labels) >= 10:
                print(f"\n>>> Đã tạo xong hình ảnh cho {len(collected_labels)} labels!")
                break


if __name__ == '__main__':
    WEIGHTS_PATH = './result/nucla/CTROGC-GCN.pt'
    analyze_weights(WEIGHTS_PATH)
