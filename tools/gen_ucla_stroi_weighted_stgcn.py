"""
Script tạo ảnh FiveFS-Weighted sử dụng weights từ ST-GCN 4-group model.
Thay vì extract feature từng sample (gây noise), script này dùng weights TĨNH
đã được học từ Edge Importance của ST-GCN.

Flow:
1. Load group_weights.json (được tạo bởi tools/train_stgcn_group.py)
   Format: {"0": {"head": 1.2, ...}, "1": {...}}
2. Load danh sách ảnh FiveFS và labels
3. Với mỗi ảnh:
   - Xác định label (action class) của ảnh đó
   - Map label -> group (dùng LABEL_TO_GROUP)
   - Lấy weights tương ứng với group đó
   - Apply weights lên 5 vùng body part
4. Lưu ảnh mới

Lợi ích:
- Weights ĐẶC THÙ cho từng nhóm hành động (vd: Group 1 pick up -> sáng tay, Group 0 walk -> sáng chân)
- Nhất quán trong cùng 1 nhóm, không bị noise
"""
import os
import cv2
import numpy as np
import json
import re
from tqdm import tqdm

# Import mapping từ feeder
import sys
sys.path.append(os.getcwd())
from feeder.feeder_nucla_group import LABEL_TO_GROUP, GROUP_NAMES

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
        group_weights = json.load(f)
    
    print(">>> Loaded Group Weights:")
    for g, weights in group_weights.items():
        print(f"  Group {g} ({GROUP_NAMES[int(g)]}): {weights}")
        
    # Chuẩn bị weight arrays cho từng group
    # Thứ tự rows: head, l_hand, r_hand, l_leg, r_leg
    parts_order = ['head', 'l_hand', 'r_hand', 'l_leg', 'r_leg']
    
    # Pre-compute weight arrays for each group
    group_weight_arrays = {}
    for g, weights in group_weights.items():
        vals = np.array([weights[p] for p in parts_order])
        
        # Normalize về [0.5, 1.5]
        w_min, w_max = vals.min(), vals.max()
        if (w_max - w_min) > 0:
            vals = 0.5 + 1.0 * (vals - w_min) / (w_max - w_min)
        else:
            vals = np.ones(5)
        
        group_weight_arrays[int(g)] = vals

    # Load data_list để biết label của từng file ảnh (từ feeder gốc hoặc logic parse filename)
    # Filename format: a01_s01_e01_v01.png
    # a01 -> action 1
    
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
            
            # Parse extracted label from filename
            # format: a10_s01_e01_v03....
            match = re.search(r'a(\d+)_', img_name)
            if not match:
                continue
            
            action_label = int(match.group(1))
            group_label = LABEL_TO_GROUP.get(action_label, 0)
            
            # Lấy weights tương ứng cho group này
            current_weights = group_weight_arrays[group_label]
            
            h, w, c = img.shape
            # Ảnh FiveFS có 5 rows tương ứng 5 body parts
            part_h = h // 5
            
            img_float = img.astype(np.float32)
            
            for i, weight in enumerate(current_weights):
                y_start = i * part_h
                y_end = (i + 1) * part_h if i < 4 else h
                
                # Apply weight
                img_float[y_start:y_end, :, :] *= weight

            # Print debug for the first few images to verify weights are applied
            # logging request: "log trong cái ảnh đi xem nó weight tới ảnh nào thì log ra"
            print(f"   > Weighted: {img_name} | Label: {action_label} -> Group: {group_label} ({GROUP_NAMES[group_label]})")
            
            # Clip và convert lại uint8
            img_weighted = np.clip(img_float, 0, 255).astype(np.uint8)
            
            # Save
            save_path = os.path.join(out_split_dir, img_name)
            cv2.imwrite(save_path, img_weighted)
            
            # User requested log: log which image is being weighted
            # To avoid spamming, maybe print every 100 images or if explicit debug flag is on? 
            # But user said "log trong cái ảnh đi", implies they want to see it. 
            # Let's print for the first few images of each batch or just use tqdm description if possible, 
            # but user specifically asked for logs. Let's print it.
            # actually printing every line might be too much for 1000s of images. 
            # I will print the first 5 and then every 100th.
            # Or just print as requested. usage of tqdm handles progress, but maybe they want to see the Group/Weight applied.
            # I'll add a detailed print for the first 5 images of each split.
            pass

    print("\n>>> Done! Weighted images saved to:", output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='./result/nucla/stgcn_group/group_weights.json', help='Path to group_weights.json')
    # Default input path might be wrong, logic below tries to detect it
    parser.add_argument('--input', default='../drive/MyDrive/Data/ucla_fivefs', help='Path to input FiveFS images')
    parser.add_argument('--output', default='./ucla_stroi_weighted_stgcn/', help='Path to save weighted images')
    args = parser.parse_args()

    # Resolve absolute paths for clarity
    weights_path = os.path.abspath(args.weights)
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    print(f"DEBUG: Weights Path: {weights_path}")
    print(f"DEBUG: Input Path:   {input_path}")
    print(f"DEBUG: Output Path:  {output_path}")

    # Check existence
    if not os.path.exists(weights_path):
        print(f"ERROR: Weights file not found at {weights_path}")
    
    if not os.path.exists(input_path):
        print(f"ERROR: Input directory not found at {input_path}")
        # Try to suggest based on common locations
        cwd = os.getcwd()
        print(f"       Current working directory: {cwd}")
        
        # Check potential locations
        candidates = [
            os.path.abspath('/ucla_stroi'),
            os.path.abspath('ucla_stroi'),
            os.path.abspath('../ucla_stroi'),
            os.path.join(cwd, 'ucla_stroi')
        ]
        
        found_candidate = None
        for cand in candidates:
            if os.path.exists(cand):
                print(f"       FOUND CANDIDATE: {cand}")
                found_candidate = cand
        
        if found_candidate:
            print(f"       >>> Using found candidate: {found_candidate}")
            input_path = found_candidate
        else:
            print("       Could not auto-locate 'ucla_stroi' or 'ucla_fivefs'. Please specify --input.")
            # We don't exit, we let it try (and fail inside function) or maybe exit here?
            # Let's let it run so the function can report 0 images if path is valid but empty, or crash if invalid.
            # actually the function checks os.path.exists(in_split_dir)
            pass
    
    generate_weighted_images_from_json(weights_path, input_path, output_path)
