"""
Skeleton-to-Image Encoding — hỗ trợ NTU-RGB+D (25 joints) VÀ NW-UCLA (20 joints).

Converts 3D skeleton data into an RGB image (224, 224, 3).

Algorithm:
  1. Joint Reordering: Reorder joints into 5 anatomical groups
  2. Channel Mapping: X -> R, Y -> G, Z -> B
  3. Normalization: Min-Max scaling to [0, 255]
  4. Resizing: Bilinear interpolation to (224, 224)

Cách dùng:
  # 1) Hàm đơn lẻ:
  from skeleton_to_image import skeleton_to_image
  img = skeleton_to_image(skeleton, dataset='ucla')   # (3, T, 20) -> (224, 224, 3)
  img = skeleton_to_image(skeleton, dataset='ntu')     # (3, T, 25) -> (224, 224, 3)

  # 2) Chạy batch cho UCLA dataset (dùng Feeder):
  python skeleton_to_image.py --dataset ucla --data_path ./data --output ./output

  # 3) Chạy từ file .npy (NTU):
  python skeleton_to_image.py --dataset ntu --input data.npy --output ./output
"""

import os
import sys
import numpy as np
import cv2
import json
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ============================================================================
# Joint Reordering Configurations
# ============================================================================

# --- NTU-RGB+D: 25 joints ---
# 0-SpineBase, 1-SpineMid, 2-Neck, 3-Head, 4-ShoulderLeft, 5-ElbowLeft,
# 6-WristLeft, 7-HandLeft, 8-ShoulderRight, 9-ElbowRight, 10-WristRight,
# 11-HandRight, 12-HipLeft, 13-KneeLeft, 14-AnkleLeft, 15-FootLeft,
# 16-HipRight, 17-KneeRight, 18-AnkleRight, 19-FootRight,
# 20-SpineShoulder, 21-HandTipLeft, 22-ThumbLeft, 23-HandTipRight, 24-ThumbRight
NTU_JOINT_REORDER = [
    # Group 1 - Spine (5 joints)
    3, 2, 20, 1, 0,
    # Group 2 - Left Arm (6 joints)
    4, 5, 6, 7, 22, 21,
    # Group 3 - Right Arm (6 joints)
    8, 9, 10, 11, 24, 23,
    # Group 4 - Left Leg (4 joints)
    12, 13, 14, 15,
    # Group 5 - Right Leg (4 joints)
    16, 17, 18, 19,
]

# --- NW-UCLA: 20 joints (Kinect v1 format, 1-indexed in paper → 0-indexed here) ---
# 0-HipCenter, 1-Spine, 2-ShoulderCenter, 3-Head,
# 4-ShoulderLeft, 5-ElbowLeft, 6-WristLeft, 7-HandLeft,
# 8-ShoulderRight, 9-ElbowRight, 10-WristRight, 11-HandRight,
# 12-HipLeft, 13-KneeLeft, 14-AnkleLeft, 15-FootLeft,
# 16-HipRight, 17-KneeRight, 18-AnkleRight, 19-FootRight
UCLA_JOINT_REORDER = [
    # Group 1 - Spine (4 joints)
    3, 2, 1, 0,
    # Group 2 - Left Arm (4 joints)
    4, 5, 6, 7,
    # Group 3 - Right Arm (4 joints)
    8, 9, 10, 11,
    # Group 4 - Left Leg (4 joints)
    12, 13, 14, 15,
    # Group 5 - Right Leg (4 joints)
    16, 17, 18, 19,
]

DATASET_CONFIG = {
    'ntu': {
        'num_joints': 25,
        'joint_reorder': NTU_JOINT_REORDER,
    },
    'ucla': {
        'num_joints': 20,
        'joint_reorder': UCLA_JOINT_REORDER,
    },
}

OUTPUT_SIZE = 224


# ============================================================================
# Core Functions
# ============================================================================

def skeleton_to_image(
    skeleton: np.ndarray,
    dataset: str = 'ucla',
    output_size: int = OUTPUT_SIZE,
) -> np.ndarray:
    """
    Encode a 3D skeleton sequence into an RGB image.

    Args:
        skeleton: numpy array of shape (3, T, V) hoặc (C, T, V, M).
                  - (3, T, 25) cho NTU-RGB+D
                  - (3, T, 20) cho NW-UCLA
                  - (3, T, V, 1) format từ GCN model cũng được hỗ trợ
        dataset: 'ntu' hoặc 'ucla'
        output_size: kích thước ảnh output (default 224).

    Returns:
        image: numpy array shape (output_size, output_size, 3), dtype uint8, RGB.
    """
    dataset = dataset.lower()
    assert dataset in DATASET_CONFIG, f"Dataset '{dataset}' không hỗ trợ. Dùng: {list(DATASET_CONFIG.keys())}"
    config = DATASET_CONFIG[dataset]

    # Handle (C, T, V, M) format from GCN models → squeeze M dimension
    if skeleton.ndim == 4:
        skeleton = skeleton[:, :, :, 0]  # lấy person đầu tiên

    assert skeleton.ndim == 3, f"Expected 3D array (C, T, V), got shape {skeleton.shape}"
    C, T, V = skeleton.shape
    assert C == 3, f"Expected 3 channels (X, Y, Z), got {C}"
    assert V == config['num_joints'], \
        f"Dataset '{dataset}' expects {config['num_joints']} joints, got {V}"

    # Step 1: Joint Reordering
    joint_order = config['joint_reorder']
    skeleton_reordered = skeleton[:, :, joint_order]  # (3, T, V_reordered)

    # Step 2: Channel Mapping → X->R, Y->G, Z->B
    channel_x = skeleton_reordered[0]  # (T, V_reordered) → R
    channel_y = skeleton_reordered[1]  # (T, V_reordered) → G
    channel_z = skeleton_reordered[2]  # (T, V_reordered) → B

    # Step 3: Min-Max Normalization to [0, 255]
    def min_max_normalize(arr):
        a_min, a_max = arr.min(), arr.max()
        if a_max - a_min < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - a_min) / (a_max - a_min) * 255.0).astype(np.float32)

    r = min_max_normalize(channel_x)
    g = min_max_normalize(channel_y)
    b = min_max_normalize(channel_z)

    # Step 4: Resize with bilinear interpolation → (output_size, output_size)
    r = cv2.resize(r, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    g = cv2.resize(g, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(b, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    # Step 5: Stack → (H, W, 3) uint8
    image = np.stack([r, g, b], axis=-1)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def skeleton_to_image_batch(
    skeletons: np.ndarray,
    dataset: str = 'ucla',
    output_size: int = OUTPUT_SIZE,
) -> np.ndarray:
    """
    Encode a batch of skeleton sequences.

    Args:
        skeletons: (N, 3, T, V) hoặc (N, 3, T, V, M)
    Returns:
        images: (N, output_size, output_size, 3), dtype uint8.
    """
    N = skeletons.shape[0]
    images = np.zeros((N, output_size, output_size, 3), dtype=np.uint8)
    for i in range(N):
        images[i] = skeleton_to_image(skeletons[i], dataset, output_size)
    return images


def save_skeleton_image(skeleton, save_path, dataset='ucla', output_size=OUTPUT_SIZE):
    """Encode skeleton rồi save ra file ảnh."""
    image = skeleton_to_image(skeleton, dataset, output_size)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    cv2.imwrite(save_path, image_bgr)


# ============================================================================
# UCLA Dataset Pipeline — load qua Feeder hoặc JSON
# ============================================================================

def generate_ucla_images(data_path, output_path, split='all', output_size=OUTPUT_SIZE):
    """
    Tạo ảnh skeleton-to-image cho toàn bộ dataset NW-UCLA.
    Tương thích với cả CTR-GCN và ST-GCN (cùng dùng 20-joint UCLA skeleton).

    Args:
        data_path: đường dẫn tới folder chứa các sample (mỗi sample là 1 folder 
                   chứa file JSON, ví dụ: data_path/a01_s01_e01/a01_s01_e01.json)
        output_path: đường dẫn lưu ảnh output
        split: 'train', 'val', hoặc 'all'
        output_size: kích thước ảnh output
    """
    try:
        from feeder.feeder_nucla_gcn import Feeder
    except ImportError:
        print("Không import được Feeder. Đang thử load trực tiếp từ JSON...")
        _generate_from_json(data_path, output_path, output_size)
        return

    splits = ['train', 'val'] if split == 'all' else [split]
    os.makedirs(output_path, exist_ok=True)
    total = 0

    for s in splits:
        print(f"\n>>> Đang xử lý split: {s}")

        # Tạo đường dẫn label phù hợp với Feeder
        # Feeder kiểm tra 'val' in label_path để xác định split
        label_path = f'{s}_label'

        feeder = Feeder(
            data_path=data_path,
            label_path=label_path,
            random_choose=False,
            random_shift=False,
            random_move=False,
            window_size=50,
        )

        print(f"   Tổng số samples: {len(feeder)}")

        for i in tqdm(range(len(feeder)), desc=f"  {s}"):
            # Feeder trả về: (data, rgb_tensor, label, index)
            data, _, label, idx = feeder[i]

            # data shape từ feeder: (3, T, 20, 1) → cần squeeze M
            if isinstance(data, np.ndarray):
                skeleton = data
            else:
                skeleton = data.numpy()

            # Lấy tên file
            sample_info = feeder.data_dict[i % len(feeder.data_dict)]
            file_name = sample_info['file_name']

            save_path = os.path.join(output_path, f"{file_name}.png")
            save_skeleton_image(skeleton, save_path, dataset='ucla', output_size=output_size)
            total += 1

    print(f"\nHoàn tất! Đã tạo {total} ảnh tại: {output_path}")


def _generate_from_json(data_path, output_path, output_size=OUTPUT_SIZE):
    """
    Fallback: load skeleton trực tiếp từ file JSON (không cần Feeder).
    Mỗi folder sample chứa file JSON với key 'skeletons'.
    """
    os.makedirs(output_path, exist_ok=True)
    total = 0

    if not os.path.isdir(data_path):
        print(f"Error: {data_path} không phải là folder")
        return

    sample_dirs = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])

    print(f"Tìm thấy {len(sample_dirs)} samples trong {data_path}")

    for sample_name in tqdm(sample_dirs, desc="Processing"):
        json_path = os.path.join(data_path, sample_name, f"{sample_name}.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        skeletons = np.array(json_data['skeletons'])  # (T, 20, 3)

        # Subtract center (spine joint) và normalize
        if skeletons.shape[0] > 0:
            center = skeletons[0, 1, :]  # Spine joint
            skeletons = skeletons - center

        # Flatten, min-max normalize, reshape
        flat = skeletons.reshape(-1, 3)
        v_min = flat.min(axis=0)
        v_max = flat.max(axis=0)
        flat = (flat - v_min) / (v_max - v_min + 1e-6)
        flat = flat * 2 - 1
        skeletons = flat.reshape(-1, 20, 3)

        # Transpose: (T, 20, 3) → (3, T, 20)
        skeleton = np.transpose(skeletons, (2, 0, 1)).astype(np.float32)

        save_path = os.path.join(output_path, f"{sample_name}.png")
        save_skeleton_image(skeleton, save_path, dataset='ucla', output_size=output_size)
        total += 1

    print(f"\nHoàn tất! Đã tạo {total} ảnh tại: {output_path}")


# ============================================================================
# NTU-RGB+D Pipeline
# ============================================================================

def generate_ntu_images(input_path, output_path, output_size=OUTPUT_SIZE):
    """
    Tạo ảnh skeleton-to-image từ file .npy chứa NTU-RGB+D skeleton data.

    Args:
        input_path: đường dẫn tới file .npy
                    shape (3, T, 25) cho single sample hoặc (N, 3, T, 25) cho batch
        output_path: đường dẫn lưu ảnh output
        output_size: kích thước ảnh
    """
    os.makedirs(output_path, exist_ok=True)

    print(f"Loading skeleton data từ: {input_path}")
    data = np.load(input_path, allow_pickle=True)
    print(f"  Shape: {data.shape}")

    if data.ndim == 3:
        # Single sample
        save_path = os.path.join(output_path, "skeleton_image.png")
        save_skeleton_image(data, save_path, dataset='ntu', output_size=output_size)
        print(f"Saved: {save_path}")

    elif data.ndim == 4:
        # Batch: (N, 3, T, 25) hoặc (N, 3, T, 25, M)
        N = data.shape[0]
        print(f"Processing {N} samples...")
        for i in tqdm(range(N)):
            save_path = os.path.join(output_path, f"skeleton_{i:05d}.png")
            save_skeleton_image(data[i], save_path, dataset='ntu', output_size=output_size)
        print(f"Hoàn tất! Đã tạo {N} ảnh tại: {output_path}")

    else:
        print(f"Error: shape không hỗ trợ {data.shape}. "
              f"Expected (3, T, 25) hoặc (N, 3, T, 25).")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Skeleton-to-Image Encoding cho NTU-RGB+D và NW-UCLA"
    )
    parser.add_argument(
        '--dataset', '-d', type=str, default='ucla',
        choices=['ntu', 'ucla'],
        help="Dataset: 'ntu' (25 joints) hoặc 'ucla' (20 joints)"
    )
    parser.add_argument(
        '--input', '-i', type=str, default=None,
        help="Path tới file .npy (NTU) hoặc folder data (UCLA)"
    )
    parser.add_argument(
        '--data_path', type=str, default=None,
        help="Path tới folder chứa skeleton JSON (UCLA). "
             "Dùng khi muốn load qua Feeder."
    )
    parser.add_argument(
        '--output', '-o', type=str, default='./skeleton_images',
        help="Output directory (default: ./skeleton_images)"
    )
    parser.add_argument(
        '--size', '-s', type=int, default=224,
        help="Output image size (default: 224)"
    )
    parser.add_argument(
        '--split', type=str, default='all',
        choices=['train', 'val', 'all'],
        help="Split cho UCLA dataset (default: all)"
    )
    args = parser.parse_args()

    if args.dataset == 'ucla':
        data_path = args.data_path or args.input
        if data_path is None:
            print("Error: Cần --data_path hoặc --input cho UCLA dataset")
            sys.exit(1)
        generate_ucla_images(data_path, args.output, args.split, args.size)

    elif args.dataset == 'ntu':
        if args.input is None:
            print("Error: Cần --input cho NTU dataset (file .npy)")
            sys.exit(1)
        generate_ntu_images(args.input, args.output, args.size)
