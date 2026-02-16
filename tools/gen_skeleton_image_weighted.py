"""
Skeleton-to-Image Encoding + Per-Joint Soft Attention từ CTR-GCN.

Tất cả trong 1 script — KHÔNG cần tạo ảnh gốc trước.
Chỉ cần skeleton data (JSON) + CTR-GCN weights (.pt).

Flow cho mỗi sample:
  1. Load skeleton từ feeder_nucla_gcn (JSON)
  2. Feed skeleton → CTR-GCN → extract per-joint importance
  3. Tạo skeleton-to-image: X→R, Y→G, Z→B → resize 224×224
  4. Tạo per-joint attention map → nhân lên ảnh
     - Joint nào CTR-GCN cho trọng số CAO → cột đó SÁNG lên
     - Joint nào trọng số THẤP → cột đó MỜ đi
  5. Lưu ảnh weighted

Cách dùng:
  python gen_skeleton_image_weighted.py \
      --weights ./result/nucla/CTROGC-GCN.pt \
      --data_path ../drive/MyDrive/Data/NWUCLA_SKE/all_sqe \
      --output ./skeleton_images_weighted
"""

import os
import sys
import torch
import numpy as np
import cv2
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.getcwd())

from models.ctrgcn import Model as CTRGCN
from feeder.feeder_nucla_gcn import Feeder

# ============================================================================
# Config
# ============================================================================
OUTPUT_SIZE = 224

output_device = 0 if torch.cuda.is_available() else 'cpu'
device = torch.device(f"cuda:{output_device}" if torch.cuda.is_available() else "cpu")

# PHẢI GIỐNG HỆT skeleton_to_image.py
UCLA_JOINT_REORDER = [
    3, 2, 1, 0,         # Spine: Head, ShldrCenter, Spine, HipCenter
    4, 5, 6, 7,          # Left Arm: Shoulder, Elbow, Wrist, Hand
    8, 9, 10, 11,        # Right Arm: Shoulder, Elbow, Wrist, Hand
    12, 13, 14, 15,      # Left Leg: Hip, Knee, Ankle, Foot
    16, 17, 18, 19,      # Right Leg: Hip, Knee, Ankle, Foot
]

JOINT_NAMES = [
    'Head', 'ShldrCtr', 'Spine', 'HipCtr',
    'ShldrL', 'ElbowL', 'WristL', 'HandL',
    'ShldrR', 'ElbowR', 'WristR', 'HandR',
    'HipL', 'KneeL', 'AnkleL', 'FootL',
    'HipR', 'KneeR', 'AnkleR', 'FootR',
]

NUM_JOINTS = len(UCLA_JOINT_REORDER)  # 20


# ============================================================================
# Core functions
# ============================================================================

def skeleton_to_image(skeleton_np, output_size=OUTPUT_SIZE):
    """
    Encode skeleton (3, T, 20, 1) hoặc (3, T, 20) → ảnh float32 (H, W, 3).
    CHƯA clip/uint8 — để apply attention trước.
    """
    if skeleton_np.ndim == 4:
        skeleton_np = skeleton_np[:, :, :, 0]

    C, T, V = skeleton_np.shape
    reordered = skeleton_np[:, :, UCLA_JOINT_REORDER]

    def norm(arr):
        a_min, a_max = arr.min(), arr.max()
        if a_max - a_min < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - a_min) / (a_max - a_min) * 255.0).astype(np.float32)

    r = cv2.resize(norm(reordered[0]), (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    g = cv2.resize(norm(reordered[1]), (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(norm(reordered[2]), (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    return np.stack([r, g, b], axis=-1)  # (H, W, 3) float32 [0, 255]


def extract_per_joint_importance(model, skeleton_tensor):
    """
    CTR-GCN → per-joint importance score.
    Input:  (1, C, T, V, M) tensor
    Output: (V,) numpy — importance cho mỗi joint
    """
    N, C, T, V, M = skeleton_tensor.size()

    with torch.no_grad():
        _, feature = model.extract_feature(skeleton_tensor)
        # (N, C_feat, T_feat, V, M)

    # L2 norm qua channel dim → (N, T_feat, V, M)
    intensity = (feature * feature).sum(dim=1) ** 0.5
    intensity = intensity.cpu().detach().numpy()

    person_idx = 0
    if M > 1 and intensity[0, :, :, 0].mean() < intensity[0, :, :, 1].mean():
        person_idx = 1

    # Mean qua temporal axis → (V,)
    joint_importance = intensity[0, :, :, person_idx].mean(axis=0)
    return joint_importance


def create_attention_map(joint_weights_reordered, output_size=OUTPUT_SIZE):
    """
    Per-joint weights → 2D attention map.
    Mỗi CỘT (joint) trong ảnh skeleton-to-image có 1 weight riêng.
    """
    attn_1d = joint_weights_reordered.reshape(1, -1).astype(np.float32)
    return cv2.resize(attn_1d, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


# ============================================================================
# Main pipeline
# ============================================================================

def generate_weighted_images(weights_path, data_path, output_path, output_size=OUTPUT_SIZE):
    """
    Tạo ảnh skeleton-to-image có per-joint soft attention từ CTR-GCN.
    """
    os.makedirs(output_path, exist_ok=True)

    # --- Khởi tạo CTR-GCN ---
    print(">>> Đang khởi tạo CTR-GCN...")
    model = CTRGCN(
        num_class=10, num_point=20, num_person=1,
        graph='graph.ucla.Graph',
        graph_args={'labeling_mode': 'spatial'},
        in_channels=3, drop_out=0, adaptive=True
    ).to(device)

    if weights_path and os.path.exists(weights_path):
        print(f">>> Loading weights: {weights_path}")
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(">>> Load thành công!")
        except Exception as e:
            print(f"!!! Lỗi load weights: {e}")
    else:
        print(f"!!! Không tìm thấy: {weights_path}")

    model.eval()

    # --- Process ---
    total_count = 0
    debug_count = 0
    DEBUG_LIMIT = 5

    for split in ['train', 'val']:
        print(f"\n>>> Split: {split}")

        # feeder_nucla_gcn dùng label_path để xác định split:
        # 'val' in label_path → val split, otherwise → train split
        label_path = f'{split}_label'

        feeder = Feeder(
            data_path=data_path,
            label_path=label_path,
            random_choose=False,
            random_shift=False,
            random_move=False,
            window_size=50,
        )

        loader = torch.utils.data.DataLoader(
            dataset=feeder, batch_size=1, shuffle=False, num_workers=0
        )

        for batch in tqdm(loader, desc=f"  {split}"):
            # feeder_nucla_gcn trả về: (data, rgb_tensor, label, index)
            data_ske, _, label, index = batch

            # data_ske shape: (1, 3, T, 20, 1) — skeleton only
            data_ske = data_ske.float().to(device)
            if data_ske.ndim == 4:
                data_ske = data_ske.unsqueeze(-1)  # thêm M dim nếu thiếu

            idx = index.item()
            file_name = feeder.data_dict[idx % len(feeder.data_dict)]['file_name']

            # --- Extract per-joint importance ---
            joint_importance = extract_per_joint_importance(model, data_ske)
            # (20,) — original joint order

            # Reorder theo UCLA_JOINT_REORDER
            ji_reordered = joint_importance[UCLA_JOINT_REORDER]

            # Normalize → [0.5, 1.5]
            w_min, w_max = ji_reordered.min(), ji_reordered.max()
            if (w_max - w_min) > 1e-8:
                joint_weights = 0.5 + 1.0 * (ji_reordered - w_min) / (w_max - w_min)
            else:
                joint_weights = np.ones(NUM_JOINTS)

            # Debug
            if debug_count < DEBUG_LIMIT:
                print(f"\n  [DEBUG] {file_name}:")
                for j in range(NUM_JOINTS):
                    bar = '█' * int(joint_weights[j] * 20)
                    print(f"    {JOINT_NAMES[j]:>10s} [{j:2d}]: {joint_weights[j]:.3f} {bar}")
                debug_count += 1

            # --- Tạo skeleton-to-image ---
            skeleton_np = data_ske[0].cpu().numpy()  # (3, T, 20, 1)
            image_float = skeleton_to_image(skeleton_np, output_size)  # (H, W, 3)

            # --- Apply per-joint attention ---
            attn_map = create_attention_map(joint_weights, output_size)  # (H, W)
            weighted_img = image_float * attn_map[:, :, np.newaxis]

            # Clip + uint8
            weighted_img = np.clip(weighted_img, 0, 255).astype(np.uint8)

            # Save (RGB → BGR for OpenCV)
            save_path = os.path.join(output_path, f"{file_name}.png")
            cv2.imwrite(save_path, cv2.cvtColor(weighted_img, cv2.COLOR_RGB2BGR))
            total_count += 1

    print(f"\n{'='*60}")
    print(f"Hoàn tất! Đã tạo {total_count} ảnh tại: {output_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Skeleton-to-Image + Per-Joint Soft Attention (CTR-GCN)"
    )
    parser.add_argument(
        '--weights', '-w', type=str,
        default='./result/nucla/CTROGC-GCN.pt',
        help='Path tới CTR-GCN weights (.pt)'
    )
    parser.add_argument(
        '--data_path', '-d', type=str,
        default='../drive/MyDrive/Data/NWUCLA_SKE/all_sqe',
        help='Path tới folder chứa skeleton JSON (mỗi sample 1 subfolder chứa .json)'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        default='./skeleton_images_weighted',
        help='Output directory'
    )
    parser.add_argument(
        '--size', '-s', type=int, default=224,
        help='Output image size (default: 224)'
    )
    args = parser.parse_args()

    generate_weighted_images(
        weights_path=args.weights,
        data_path=args.data_path,
        output_path=args.output,
        output_size=args.size,
    )
