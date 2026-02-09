import os
import sys
import numpy as np
import pickle
import rarfile
import io
import re
from PIL import Image
from tqdm import tqdm

SKELETON_PKL_PATH = 'nwucla_yolo_2d_skeletons.pkl'
RAR_PATH = 'NW-UCLA-ALL.rar'
OUTPUT_PATH = '/ucla_stroi'

# Cấu hình khớp xương (YOLO/COCO format - 17 keypoints)
# sửa lại index nếu file pkl dùng chuẩn khác (ví dụ OpenPose 25 điểm)
JOINTS_MAP = {
    'head': 0,       # Nose
    'l_hand': 9,     # Left Wrist
    'r_hand': 10,    # Right Wrist
    'l_leg': 15,     # Left Ankle
    'r_leg': 16      # Right Ankle
}

PART_SIZE = 96
HALF_SIZE = PART_SIZE // 2


def get_image_from_rar(rar_obj, file_path):
    """Đọc ảnh từ file rar vào bộ nhớ và convert sang PIL Image"""
    try:
        img_bytes = rar_obj.read(file_path)
        return Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        # print(f"Error reading {file_path}: {e}")
        return None

def crop_part(image_pil, center_x, center_y, half_size):
    """Cắt vùng ảnh quanh tâm (x,y), tự động pad đen nếu ra ngoài biên"""
    img_w, img_h = image_pil.size
    x1 = int(center_x - half_size)
    y1 = int(center_y - half_size)
    part_img = Image.new('RGB', (half_size*2, half_size*2), (0, 0, 0))
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(img_w, x1 + half_size*2)
    crop_y2 = min(img_h, y1 + half_size*2)
    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        cropped = image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        paste_x = crop_x1 - x1
        paste_y = crop_y1 - y1
        part_img.paste(cropped, (paste_x, paste_y))

    return part_img

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
        file_map[k].sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)) if re.search(r'(\d+)', os.path.basename(x)) else x)

    print(f"Đã index xong {len(file_map)} videos trong RAR.")
    return file_map

def process_all_samples():
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
    print("Bắt đầu tạo ảnh Fivefs...")
    count = 0

    for sample_name, skeleton_data in tqdm(all_skeletons.items()):
        # Tìm danh sách file ảnh trong rar map
        # Nếu tên trong pkl không khớp hoàn toàn với tên folder trong rar, bạn cần mapping lại ở đây
        # Ví dụ: skeleton tên 'a01_s01_e01' -> tìm key tương ứng trong rar_file_map
        if sample_name not in rar_file_map:
            # Fallback: thử tìm key nào chứa sample_name
            # (Phần này tùy vào dữ liệu của bạn, có thể bỏ qua nếu tên đã khớp)
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
        if len(data_np.shape) == 4:
             data_np = data_np[0]
        if len(data_np.shape) == 3 and data_np.shape[0] in [2, 3]:
             data_np = np.transpose(data_np, (1, 2, 0))
        temporal_rgb_frames = 5
        indices = []
        if total_frames < temporal_rgb_frames:
             indices = list(range(total_frames)) + [total_frames-1] * (temporal_rgb_frames - total_frames)
        else:
             step = total_frames / temporal_rgb_frames
             indices = [int(i * step) for i in range(temporal_rgb_frames)]
        final_canvas = Image.new('RGB', (PART_SIZE * temporal_rgb_frames, PART_SIZE * 5), (0, 0, 0))

        valid_sample = False
        for i, frame_idx in enumerate(indices):
            if frame_idx >= len(sample_files): frame_idx = len(sample_files) - 1
            img_file_path = sample_files[frame_idx]
            frame_img = get_image_from_rar(rf, img_file_path)
            if frame_img is None:
                continue
            skel_idx = min(frame_idx, data_np.shape[0] - 1)
            pose = data_np[skel_idx]
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
                frame_strip.paste(cropped_part, (0, p_idx * PART_SIZE))
            final_canvas.paste(frame_strip, (i * PART_SIZE, 0))
            valid_sample = True
        if valid_sample:
            save_name = os.path.join(OUTPUT_PATH, sample_name + '.png')
            final_canvas.save(save_name)
            count += 1

    rf.close()
    print(f"\nHoàn tất! Đã tạo {count} ảnh fivefs tại: {OUTPUT_PATH}")

if __name__ == "__main__":
    process_all_samples()