import torch
import torch.nn as nn
import numpy as np
import sys
import os
from tqdm import tqdm
from PIL import Image

sys.path.append(os.getcwd())
from models.ctrgcn import Model as CTRGCN
from feeder.feeder_nucla_fusion import Feeder

output_device = 0 if torch.cuda.is_available() else 'cpu'
device = torch.device(f"cuda:{output_device}" if torch.cuda.is_available() else "cpu")

# ========== CẤU HÌNH ==========
NUM_BODY_PARTS = 5
PART_SIZE = 96  # Phải khớp với gen_ucla_stroi.py
# Joints tương ứng 5 body parts: head, l_hand, r_hand, l_leg, r_leg (UCLA 20-joint)
TARGET_JOINTS = [3, 11, 7, 18, 14]


def generate_weighted_images(weights_path, input_fivefs_path, output_path):
    """
    Tạo ảnh FiveFS có trọng số từ ảnh FiveFS gốc.
    
    Quy trình:
    1. Load skeleton data qua Feeder → đưa vào CTR-GCN để extract feature importance
    2. Load ảnh FiveFS gốc đã tạo bởi gen_ucla_stroi.py
    3. Tính trọng số cho mỗi body part dựa trên feature importance
    4. Nhân trọng số vào từng vùng body part tương ứng trên ảnh
    5. Lưu ảnh với clip [0, 255] thay vì re-normalize
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # --- Khởi tạo mô hình CTR-GCN ---
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
        except Exception as e:
            print(f"!!! Cảnh báo: Không load được weights ({e}). Dùng weights ngẫu nhiên.")
    else:
        print("!!! Cảnh báo: Không tìm thấy weights path. Đang chạy với weights ngẫu nhiên.")

    model_ske.eval()

    splits = ['train', 'val']
    total_count = 0
    skip_count = 0
    
    for split in splits:
        print(f"\n>>> Đang xử lý tập dữ liệu: {split}")
        feeder = Feeder(
            split=split, 
            random_choose=False, 
            random_shift=False, 
            random_move=False,
            window_size=50,
            temporal_rgb_frames=5
        )
        
        loader = torch.utils.data.DataLoader(
            dataset=feeder,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        for i, (data, label, index) in enumerate(tqdm(loader)):
            data_ske = data[0].float().to(device)
            
            # Lấy tên file
            idx = index.item()
            file_name = feeder.data_dict[idx]['file_name']
            
            # --- Load ảnh FiveFS gốc ---
            fivefs_img_path = os.path.join(input_fivefs_path, file_name + '.png')
            if not os.path.exists(fivefs_img_path):
                skip_count += 1
                continue
            
            fivefs_img = Image.open(fivefs_img_path).convert('RGB')
            fivefs_np = np.array(fivefs_img).astype(np.float32)  # (H, W, 3), [0, 255]
            img_h, img_w = fivefs_np.shape[:2]
            
            # --- Extract feature importance từ skeleton ---
            N, C, T, V, M = data_ske.size()
            
            with torch.no_grad():
                _, feature = model_ske.extract_feature(data_ske)
                
            # feature shape: (N, C_feat, T_feat, V)
            intensity_s = (feature * feature).sum(dim=1) ** 0.5
            intensity_s = intensity_s.cpu().detach().numpy()
            feature_s = np.abs(intensity_s)
            
            # Normalize feature về [0, 1]
            feat_min, feat_max = feature_s.min(), feature_s.max()
            if (feat_max - feat_min) > 0:
                feature_s = (feature_s - feat_min) / (feat_max - feat_min)
            
            # --- Tính trọng số cho mỗi body part ---
            weights_per_part = np.ones(NUM_BODY_PARTS)
            
            n = 0  # batch_size = 1
            person_idx = 0
            if M > 1:
                if feature_s[n, :, :, 0].mean() < feature_s[n, :, :, 1].mean():
                    person_idx = 1
            
            temporal_positions = 15  # số timestep dùng để tính trung bình
            
            for j, v_idx in enumerate(TARGET_JOINTS):
                if v_idx < V:
                    feature_val = feature_s[n, :, v_idx, person_idx]
                    # Lấy top-k temporal positions có giá trị cao nhất
                    k = min(temporal_positions, len(feature_val))
                    top_k_vals = np.partition(feature_val, -k)[-k:]
                    weights_per_part[j] = top_k_vals.mean()
            
            # Normalize trọng số về khoảng [0.3, 1.0]
            # - Body part quan trọng nhất → giữ nguyên brightness (1.0)
            # - Body part ít quan trọng → giảm brightness (0.3)
            # → Không có giá trị > 1.0 nên không bị clip mất thông tin
            w_min, w_max = weights_per_part.min(), weights_per_part.max()
            if (w_max - w_min) > 0:
                weights_per_part = 0.3 + 0.7 * (weights_per_part - w_min) / (w_max - w_min)
            else:
                weights_per_part = np.ones(NUM_BODY_PARTS)
            
            # --- Apply trọng số lên từng vùng body part trong ảnh ---
            # Cấu trúc ảnh FiveFS:
            #   Hàng: 5 body parts (head, l_hand, r_hand, l_leg, r_leg), mỗi hàng cao PART_SIZE
            #   Cột: 5 temporal frames, mỗi cột rộng PART_SIZE
            part_h = img_h // NUM_BODY_PARTS
            
            weighted_img = fivefs_np.copy()
            for j in range(NUM_BODY_PARTS):
                y_start = j * part_h
                y_end = (j + 1) * part_h if j < NUM_BODY_PARTS - 1 else img_h
                weighted_img[y_start:y_end, :, :] *= weights_per_part[j]
            
            # Clip về [0, 255] — KHÔNG re-normalize để giữ hiệu ứng trọng số
            weighted_img = np.clip(weighted_img, 0, 255).astype(np.uint8)
            
            # --- Lưu ảnh ---
            save_full_path = os.path.join(output_path, f"{file_name}.png")
            Image.fromarray(weighted_img).save(save_full_path)
            total_count += 1

    print(f"\n{'='*50}")
    print(f"Hoàn tất! Đã tạo {total_count} ảnh weighted tại: {output_path}")
    if skip_count > 0:
        print(f"Bỏ qua {skip_count} mẫu (không tìm thấy ảnh FiveFS gốc)")


if __name__ == '__main__':
    WEIGHTS_PATH = './result/nucla/CTROGC-GCN.pt' 
    INPUT_FIVEFS_PATH = './ucla_stroi/'       # Đường dẫn tới ảnh FiveFS gốc (output của gen_ucla_stroi.py)
    OUTPUT_PATH = './ucla_stroi_weighted/'     # Đường dẫn lưu ảnh weighted
    
    generate_weighted_images(WEIGHTS_PATH, INPUT_FIVEFS_PATH, OUTPUT_PATH)
