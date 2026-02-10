import os
import sys
import numpy as np
import pickle
import rarfile
import io
import re
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.ctrgcn import Model as CTRGCN
SKELETON_PKL_PATH = 'nwucla_yolo_2d_skeletons.pkl'
RAR_PATH = 'NW-UCLA-ALL.rar'
OUTPUT_PATH = '/ucla_stroi_weighted'
WEIGHTS_PATH = './result/nucla/CTROGC-GCN.pt'
MODEL_NUM_CLASS = 10
MODEL_NUM_POINT = 20
MODEL_NUM_PERSON = 1
MODEL_GRAPH = 'graph.ucla.Graph'
MODEL_GRAPH_ARGS = {'labeling_mode': 'spatial'}
MODEL_IN_CHANNELS = 3
JOINTS_MAP = {
    'head': 0,       # Nose
    'l_hand': 9,     # Left Wrist
    'r_hand': 10,    # Right Wrist
    'l_leg': 15,     # Left Ankle
    'r_leg': 16      # Right Ankle
}
TARGET_JOINTS_FOR_WEIGHT = [3, 11, 7, 18, 14]

PART_SIZE = 96
HALF_SIZE = PART_SIZE // 2
TEMPORAL_RGB_FRAMES = 5
TEMPORAL_POSITIONS = 15
output_device = 0 if torch.cuda.is_available() else 'cpu'
device = torch.device(f"cuda:{output_device}" if torch.cuda.is_available() else "cpu")


def load_ctrgcn_model(weights_path):
    """Khởi tạo và load trọng số mô hình CTR-GCN"""
    print(">>> Đang khởi tạo mô hình CTR-GCN...")
    model = CTRGCN(
        num_class=MODEL_NUM_CLASS,
        num_point=MODEL_NUM_POINT,
        num_person=MODEL_NUM_PERSON,
        graph=MODEL_GRAPH,
        graph_args=MODEL_GRAPH_ARGS,
        in_channels=MODEL_IN_CHANNELS
    ).to(device)

    if weights_path and os.path.exists(weights_path):
        print(f">>> Loading weights từ: {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(">>> Load weights thành công!")
        except Exception as e:
            print(f"!!! Cảnh báo: Không load được weights ({e}). Dùng weights ngẫu nhiên.")
    else:
        print("!!! Cảnh báo: Không tìm thấy weights path. Đang chạy với weights ngẫu nhiên.")

    model.eval()
    return model


def compute_joint_weights(model, skeleton_data_for_model):
    """
    Tính trọng số cho từng khớp dựa trên feature attention từ CTR-GCN.
    
    Input:
        model: CTR-GCN model đã load weights
        skeleton_data_for_model: numpy array shape (C, T, V, M) - dữ liệu skeleton
                                 đã chuẩn hóa cho model (C=3, T=frames, V=20, M=1)
    
    Output:
        weight_per_joint: dict {joint_index: weight_value} cho 5 khớp mục tiêu
    """
    x = torch.tensor(skeleton_data_for_model).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        _, feature = model.extract_feature(x)
    
    intensity = (feature * feature).sum(dim=1) ** 0.5
    intensity = intensity.cpu().detach().numpy()
    feature_s = np.abs(intensity)
    
    if (feature_s.max() - feature_s.min()) != 0:
        feature_s = 255 * (feature_s - feature_s.min()) / (feature_s.max() - feature_s.min())
    
    N, T_out, V, M = feature_s.shape  # N=1
    n = 0
    
    person_idx = 0
    if M > 1:
        if feature_s[n, :, :, 0].mean() < feature_s[n, :, :, 1].mean():
            person_idx = 1
    
    weight_per_joint = {}
    for j, v in enumerate(TARGET_JOINTS_FOR_WEIGHT):
        if v < V:
            feature_val = feature_s[n, :, v, person_idx]
            k = min(TEMPORAL_POSITIONS, len(feature_val) - 1)
            temp = np.partition(-feature_val, k)
            avg_feat = -temp[:TEMPORAL_POSITIONS].mean()
            weight_per_joint[j] = avg_feat / 127.0
        else:
            weight_per_joint[j] = 1.0
    
    return weight_per_joint


def get_image_from_rar(rar_obj, file_path):
    """Đọc ảnh từ file rar vào bộ nhớ và convert sang PIL Image"""
    try:
        img_bytes = rar_obj.read(file_path)
        return Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return None


def crop_part(image_pil, center_x, center_y, half_size):
    """Cắt vùng ảnh quanh tâm (x,y), tự động pad đen nếu ra ngoài biên"""
    img_w, img_h = image_pil.size
    x1 = int(center_x - half_size)
    y1 = int(center_y - half_size)
    part_img = Image.new('RGB', (half_size * 2, half_size * 2), (0, 0, 0))
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(img_w, x1 + half_size * 2)
    crop_y2 = min(img_h, y1 + half_size * 2)
    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        cropped = image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        paste_x = crop_x1 - x1
        paste_y = crop_y1 - y1
        part_img.paste(cropped, (paste_x, paste_y))
    return part_img


def apply_weight_to_part(part_img, weight_value):
    """
    Nhân trọng số attention vào ảnh part (giống visual.py: rgb * weight).
    Weight_value là scalar float.
    """
    img_np = np.array(part_img).astype(np.float32)
    img_np = img_np * weight_value
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def build_rar_index(rar_obj):
    """
    Tạo index để tìm file trong RAR nhanh hơn.
    Map: tên_sample -> list các đường dẫn file ảnh trong rar
    """
    print("Đang tạo index cho file RAR (việc này có thể mất vài giây)...")
    file_map = {}
    for f in rar_obj.namelist():
        if f.lower().endswith(('.jpg', '.png')):
            parts = f.split('/')
            if len(parts) >= 2:
                sample_name = parts[-2]
                if sample_name not in file_map:
                    file_map[sample_name] = []
                file_map[sample_name].append(f)
    print("Đang sắp xếp frame...")
    for k in file_map:
        file_map[k].sort(
            key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1))
            if re.search(r'(\d+)', os.path.basename(x)) else x
        )
    print(f"Đã index xong {len(file_map)} videos trong RAR.")
    return file_map


def prepare_skeleton_for_model(data_np, window_size=50):
    """
    Chuẩn bị dữ liệu skeleton theo format CTR-GCN cần: (C, T, V, M).
    Input data_np: shape linh hoạt, cần chuyển về (C=3, T, V=20, M=1).
    """
    if len(data_np.shape) == 4:
        if data_np.shape[0] in [1, 2] and data_np.shape[-1] in [2, 3]:
            data_np = data_np[0]
    
    if len(data_np.shape) == 3:
        if data_np.shape[0] in [2, 3] and data_np.shape[-1] != 2 and data_np.shape[-1] != 3:
            data_np = data_np[:, :, :, np.newaxis]
        elif data_np.shape[-1] in [2, 3]:
            data_np = np.transpose(data_np, (2, 0, 1))
            data_np = data_np[:, :, :, np.newaxis]
        else:
            data_np = data_np[:, :, :, np.newaxis]
    
    C, T, V, M = data_np.shape
    if C == 2:
        zeros = np.zeros((1, T, V, M))
        data_np = np.concatenate([data_np, zeros], axis=0)
    
    C, T, V, M = data_np.shape
    if T < window_size:
        pad = np.zeros((C, window_size - T, V, M))
        data_np = np.concatenate([data_np, pad], axis=1)
    elif T > window_size:
        indices = np.linspace(0, T - 1, window_size, dtype=int)
        data_np = data_np[:, indices, :, :]
    
    return data_np


def process_all_samples():
    model = load_ctrgcn_model(WEIGHTS_PATH)

    print(f"Loading skeletons from {SKELETON_PKL_PATH}...")
    with open(SKELETON_PKL_PATH, 'rb') as f:
        all_skeletons = pickle.load(f)
    
    if not os.path.exists(RAR_PATH):
        print(f"Không tìm thấy file RAR: {RAR_PATH}")
        return

    rf = rarfile.RarFile(RAR_PATH)
    rar_file_map = build_rar_index(rf)
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    print("Bắt đầu tạo ảnh Fivefs có trọng số...")
    count = 0

    for sample_name, skeleton_data in tqdm(all_skeletons.items()):
        if sample_name not in rar_file_map:
            found = False
            for k in rar_file_map:
                if sample_name in k:
                    sample_files = rar_file_map[k]
                    found = True
                    break
            if not found:
                continue
        else:
            sample_files = rar_file_map[sample_name]

        total_frames = len(sample_files)
        if total_frames == 0:
            continue

        if isinstance(skeleton_data, dict) and 'keypoint' in skeleton_data:
            data_np = np.array(skeleton_data['keypoint'])
        else:
            data_np = np.array(skeleton_data)
        
        data_for_crop = data_np.copy()
        if len(data_for_crop.shape) == 4:
            data_for_crop = data_for_crop[0]
        if len(data_for_crop.shape) == 3 and data_for_crop.shape[0] in [2, 3]:
            data_for_crop = np.transpose(data_for_crop, (1, 2, 0))

        try:
            data_for_model = prepare_skeleton_for_model(data_np.copy())
            joint_weights = compute_joint_weights(model, data_for_model)
        except Exception as e:
            joint_weights = {j: 1.0 for j in range(5)}

        indices = []
        if total_frames < TEMPORAL_RGB_FRAMES:
            indices = list(range(total_frames)) + [total_frames - 1] * (TEMPORAL_RGB_FRAMES - total_frames)
        else:
            step = total_frames / TEMPORAL_RGB_FRAMES
            indices = [int(i * step) for i in range(TEMPORAL_RGB_FRAMES)]

        final_canvas = Image.new('RGB', (PART_SIZE * TEMPORAL_RGB_FRAMES, PART_SIZE * 5), (0, 0, 0))

        valid_sample = False
        for i, frame_idx in enumerate(indices):
            if frame_idx >= len(sample_files):
                frame_idx = len(sample_files) - 1
            img_file_path = sample_files[frame_idx]
            frame_img = get_image_from_rar(rf, img_file_path)
            if frame_img is None:
                continue

            skel_idx = min(frame_idx, data_for_crop.shape[0] - 1)
            pose = data_for_crop[skel_idx]

            frame_strip = Image.new('RGB', (PART_SIZE, PART_SIZE * 5), (0, 0, 0))
            parts_order = ['head', 'l_hand', 'r_hand', 'l_leg', 'r_leg']

            for p_idx, part_name in enumerate(parts_order):
                joint_idx = JOINTS_MAP.get(part_name)
                if joint_idx >= pose.shape[0]:
                    continue

                cx, cy = pose[joint_idx][0], pose[joint_idx][1]

                if cx <= 1 and cy <= 1:
                    continue

                cropped_part = crop_part(frame_img, cx, cy, HALF_SIZE)
                
                w = joint_weights.get(p_idx, 1.0)
                weighted_part = apply_weight_to_part(cropped_part, w)
                
                frame_strip.paste(weighted_part, (0, p_idx * PART_SIZE))

            final_canvas.paste(frame_strip, (i * PART_SIZE, 0))
            valid_sample = True

        if valid_sample:
            save_name = os.path.join(OUTPUT_PATH, sample_name + '.png')
            final_canvas.save(save_name)
            count += 1

    rf.close()
    print(f"\nHoàn tất! Đã tạo {count} ảnh fivefs có trọng số tại: {OUTPUT_PATH}")


if __name__ == "__main__":
    process_all_samples()
