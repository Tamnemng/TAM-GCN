import os
import sys
import numpy as np
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm

sys.path.append('./')
from models.ctrgcn import Model as CTRGCNModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_WEIGHTS = './result/nucla/CTROGC-GCN.pt'

PART_SIZE = 96
# Gom nhóm 20 khớp của bộ dữ liệu NW-UCLA thành 5 phần cơ thể chính (0-indexed)
BODY_PARTS = {
    'head_torso': [0, 1, 2, 3], # Hip Center, Spine, Shoulder Center, Head
    'l_hand': [4, 5, 6, 7],     # Left Shoulder, Elbow, Wrist, Hand
    'r_hand': [8, 9, 10, 11],   # Right Shoulder, Elbow, Wrist, Hand
    'l_leg': [12, 13, 14, 15],  # Left Hip, Knee, Ankle, Foot
    'r_leg': [16, 17, 18, 19]   # Right Hip, Knee, Ankle, Foot
}

PART_NAMES = list(BODY_PARTS.keys())

def load_3d_skeleton(sample_name, json_data_dir):
    # Lấy 3D skeleton từ thư mục json chuẩn cho GCN
    json_file = os.path.join(json_data_dir, sample_name, sample_name + ".json")
    if not os.path.exists(json_file):
        return None
    try:
        with open(json_file, 'r') as f:
            v_info = json.load(f)
        if 'skeletons' in v_info:
            skel_data = np.array(v_info['skeletons'])
        elif 'data' in v_info:
            skel_data = np.array(v_info['data'])
        else:
            return None, None
        
        if len(skel_data.shape) == 2:
            T, _ = skel_data.shape
            skel_data = skel_data.reshape(T, 20, 3)
            
        # (T, V, C) -> (C, T, V)
        data_numpy = skel_data.transpose(2, 0, 1) 
        # (C, T, V, M)
        data_numpy = np.expand_dims(data_numpy, axis=-1) 
        
        # padding/trimming to exactly 52 frames cho GCN
        C, T_orig, V, M = data_numpy.shape
        window_size = 52
        if T_orig < window_size:
            pad = np.zeros((C, window_size - T_orig, V, M))
            data_numpy = np.concatenate((data_numpy, pad), axis=1)
        elif T_orig > window_size:
            begin = (T_orig - window_size) // 2
            data_numpy = data_numpy[:, begin:begin+window_size, :, :]
            
        return data_numpy, T_orig
    except Exception as e:
        return None, None

def extract_spatio_temporal_weights(model, x_3d):
    x_tensor = torch.tensor(x_3d).unsqueeze(0).float().to(device)
    with torch.no_grad():
        # Trích xuất Feature Map từ layer cuối cùng (chứa cả Spatial & Temporal info)
        feat, _ = model.extract_feature(x_tensor)
        
    feat_abs = feat.abs()
    weight_tv = feat_abs.mean(dim=(0, 1, 4)).squeeze() 
    weight_tv = weight_tv.cpu().numpy()
    
    T = weight_tv.shape[0]
    
    # Gom 20 khớp thành ma trận [T, 5] 
    part_weights = np.zeros((T, 5))
    for t in range(T):
        for i, p_name in enumerate(PART_NAMES):
            idx = BODY_PARTS[p_name]
            part_weights[t, i] = np.max(weight_tv[t, idx])
            
    # Áp dụng logarithm để giảm biên độ đỉnh
    part_weights = np.log1p(part_weights)
    return part_weights

def apply_soft_weighting(image, weights, T_orig):
    # weights shape: [T, 5] (thường T=52)
    weights_img = weights.T 
    _, T = weights_img.shape
    
    num_frames_stroi = 5
    w_5x5 = np.zeros((5, num_frames_stroi))
    
    step = T_orig / num_frames_stroi
    stroi_indices = [int(i * step) for i in range(num_frames_stroi)]
    
    for i, orig_idx in enumerate(stroi_indices):
        if T_orig < 52:
            gcn_idx = orig_idx
        else:
            begin = (T_orig - 52) // 2
            gcn_idx = orig_idx - begin
            
        gcn_idx = max(0, min(T - 1, gcn_idx))
        
        idx_start = max(0, gcn_idx - 2)
        idx_end = min(T, gcn_idx + 3)
        w_5x5[:, i] = np.max(weights_img[:, idx_start:idx_end], axis=1)

    # TIẾN HÀNH CHUẨN HOÁ TRỰC TIẾP TRÊN MA TRẬN 5x5 NÀY
    # Đảm bảo rằng trong 25 ô trên ảnh, LUÔN CÓ 1 ô sáng nhất (1.0) và 1 ô tối nhất
    w_min = np.min(w_5x5)
    w_max = np.max(w_5x5)
    
    if w_max > w_min:
        w_5x5 = (w_5x5 - w_min) / (w_max - w_min)
    else:
        w_5x5 = np.ones((5, num_frames_stroi))
        
    # Tăng cường độ mờ thấp nhất từ 0.3 lên 0.45 để ResNet50 vẫn thấy rõ bối cảnh các quỹ đạo gốc
    MIN_INTENSITY = 0.45
    w_5x5 = w_5x5 * (1.0 - MIN_INTENSITY) + MIN_INTENSITY
            
    # Phóng to ma trận trọng số 5x5 lên kích thước pixel ảnh bằng NEAREST để giữ nguyên các ô Grid
    w_image = Image.fromarray(np.uint8(w_5x5 * 255), mode='L')
    w_mask = w_image.resize((image.width, image.height), resample=Image.NEAREST)
    
    # Bước 3: Áp Mask vào ảnh chuẩn
    img_np = np.array(image.convert('RGB')).astype(np.float32)
    mask_np = np.array(w_mask).astype(np.float32) / 255.0
    mask_np = np.expand_dims(mask_np, axis=2)
    
    out_np = np.clip(img_np * mask_np, 0, 255).astype(np.uint8)
    return Image.fromarray(out_np)

def process_all(input_dir, output_dir, json_data_dir):
    if not os.path.exists(input_dir):
        print(f"LỖI: Thư mục chứa ảnh Stroi '{input_dir}' không tồn tại!")
        print("Hãy chạy gen_ucla_stroi.py trước hoặc cung cấp đúng đường dẫn với biến --input_dir")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nKhởi tạo mô hình CTR-GCN để trích xuất Spatio-Temporal Feature Map...")
    model = CTRGCNModel(
        num_class=10, num_point=20, num_person=1, 
        graph='graph.ucla.Graph', graph_args={'labeling_mode': 'spatial'}, 
        in_channels=3
    )
    if os.path.exists(MODEL_WEIGHTS):
        weights = torch.load(MODEL_WEIGHTS, map_location=device)
        model.load_state_dict(weights)
        print(f"Đã load trọng số từ {MODEL_WEIGHTS}")
    else:
        print(f"Cảnh báo: Không tìm thấy {MODEL_WEIGHTS}. Sẽ dùng trọng số khởi tạo ngẫu nhiên.")
        
    model.eval()
    model = model.to(device)

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png') or f.endswith('.jpg')]
    if len(image_files) == 0:
        print(f"Không tìm thấy ảnh .png nào trong thư mục {input_dir}")
        return
        
    print(f"\nBắt đầu lướt qua {len(image_files)} ảnh đã tạo và áp dụng Soft Masking...")
    
    count = 0
    tbar = tqdm(image_files)
    for img_file in tbar:
        sample_name = os.path.splitext(img_file)[0]
        
        # 1. Load ảnh Stroi gốc
        img_path = os.path.join(input_dir, img_file)
        try:
            img = Image.open(img_path)
            if img.height < 5 * PART_SIZE:
                continue
        except Exception as e:
            continue
            
        # 2. Xử lý Skeleton vào GCN
        x_3d, T_orig = load_3d_skeleton(sample_name, json_data_dir)
        if x_3d is not None:
            # Lấy ma trận trọng số [T, 5]
            weights = extract_spatio_temporal_weights(model, x_3d)
            
            # Áp dụng nhân soft masking (Làm tối ảnh mượt mà) thay vì tô màu vàng
            weighted_img = apply_soft_weighting(img, weights, T_orig)
            
            save_path = os.path.join(output_dir, img_file)
            weighted_img.save(save_path)
            count += 1
            
    print(f"\nHoàn tất! Đã lưu {count} ảnh Soft Weighting STROI vào:\n ---> {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Áp dụng Soft Weighting từ CTR-GCN lên ảnh STROI")
    parser.add_argument('--input_dir', type=str, default='/ucla_stroi', help='Thư mục chứa ảnh 5x5 Stroi đầu vào (Ảnh gốc chưa nhuộm)')
    parser.add_argument('--output_dir', type=str, default='./ucla_stroi_softmap', help='Thư mục chứa ảnh kết quả đã làm mờ soft weighting')
    parser.add_argument('--JSON_DATA_DIR', type=str, default=r'C:\Users\nguyn\Downloads\NW-UCLA-ALL\NW-UCLA-ALL', help='Thư mục chứa file JSON 3D Skeletons của UCLA')
    args = parser.parse_args()
    
    process_all(args.input_dir, args.output_dir, args.JSON_DATA_DIR)
