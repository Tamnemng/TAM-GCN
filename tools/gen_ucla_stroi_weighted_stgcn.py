"""
Script tạo ảnh FiveFS-Weighted sử dụng weights từ ST-GCN 4-group model.
Thay vì extract feature từng sample (gây noise), script này dùng weights TĨNH
đã được học từ Edge Importance của ST-GCN.

Flow:
1. Load body_part_weights.json (được tạo bởi tools/train_stgcn_group.py)
   Format: {"head": 1.2, "l_hand": 0.8, ...}
2. Load ảnh FiveFS gốc
3. Apply weights lên từng vùng ảnh tương ứng với body part
4. Lưu ảnh mới

Lợi ích:
- Weights nhất quán, không bị noise theo từng sample
- Phản ánh đúng độ quan trọng GLOBAL của từng body part đối với việc phân loại action
"""
import os
import cv2
import numpy as np
import json
from tqdm import tqdm

def generate_weighted_images_from_json(weights_path, input_path, output_path):
    print("=" * 60)
    print("GENERATING WEIGHTED IMAGES (ST-GCN METHOD)")
    print("=" * 60)
    
    # 1. Load weights
    if not os.path.exists(weights_path):
        print(f"!!! Error: Không tìm thấy file weights {weights_path}")
        print("Hãy chạy tools/train_stgcn_group.py trước để tạo file này!")
        return

    with open(weights_path, 'r') as f:
        body_part_weights = json.load(f)
    
    print(">>> Loaded Body Part Weights:")
    for part, w in body_part_weights.items():
        print(f"  {part:>8s}: {w:.4f}")
        
    # Chuẩn bị array weights theo thứ tự rows của ảnh FiveFS
    # Thứ tự rows: head, l_hand, r_hand, l_leg, r_leg
    parts_order = ['head', 'l_hand', 'r_hand', 'l_leg', 'r_leg']
    weight_vals = np.array([body_part_weights[p] for p in parts_order])
    
    # Normalize weights về [0.5, 1.5]
    # (Optional: nếu weights từ ST-GCN đã OK thì có thể không cần normalize, 
    #  nhưng để an toàn ta vẫn map về range này để tránh làm ảnh quá tối/sáng)
    w_min, w_max = weight_vals.min(), weight_vals.max()
    if (w_max - w_min) > 0:
        weight_vals = 0.5 + 1.0 * (weight_vals - w_min) / (w_max - w_min)
    else:
        weight_vals = np.ones(5)
        
    print(f">>> Normalized Weights applied to images: {dict(zip(parts_order, weight_vals.round(3)))}")

    # 2. Process images
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Lấy danh sách file ảnh
    # Cấu trúc folder FiveFS: train/ và val/
    for split in ['train', 'val']:
        in_split_dir = os.path.join(input_path, split)
        out_split_dir = os.path.join(output_path, split)
        
        if not os.path.exists(in_split_dir):
            continue
            
        os.makedirs(out_split_dir, exist_ok=True)
        
        img_list = [f for f in os.listdir(in_split_dir) if f.endswith('.jpg') or f.endswith('.png')]
        print(f"\nProcessing {split}: {len(img_list)} images...")
        
        for img_name in tqdm(img_list):
            img_path = os.path.join(in_split_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            h, w, c = img.shape
            # Ảnh FiveFS có 5 rows tương ứng 5 body parts
            part_h = h // 5
            
            img_float = img.astype(np.float32)
            
            for i, weight in enumerate(weight_vals):
                y_start = i * part_h
                y_end = (i + 1) * part_h if i < 4 else h
                
                # Apply weight
                img_float[y_start:y_end, :, :] *= weight
            
            # Clip và convert lại uint8
            img_weighted = np.clip(img_float, 0, 255).astype(np.uint8)
            
            # Save
            save_path = os.path.join(out_split_dir, img_name)
            cv2.imwrite(save_path, img_weighted)

    print("\n>>> Done! Weighted images saved to:", output_path)


if __name__ == '__main__':
    # Đường dẫn file weights tạo ra từ train_stgcn_group.py
    WEIGHTS_JSON = './result/nucla/stgcn_group/body_part_weights.json' 
    
    # Ảnh FiveFS gốc (đã tạo từ gen_ucla_stroi.py)
    INPUT_FIVEFS_PATH = '../drive/MyDrive/Data/ucla_fivefs'       
    
    # Output
    OUTPUT_PATH = './ucla_stroi_weighted_stgcn/'     
    
    generate_weighted_images_from_json(WEIGHTS_JSON, INPUT_FIVEFS_PATH, OUTPUT_PATH)
