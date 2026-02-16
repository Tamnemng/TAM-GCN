"""
Apply Soft Attention (per-joint) lên ảnh skeleton-to-image sử dụng CTR-GCN.

Ý tưởng:
  - Trong ảnh skeleton-to-image, trục ngang (width) = joints (sau reorder),
    trục dọc (height) = temporal frames.
  - CTR-GCN extract feature importance cho TỪNG JOINT riêng lẻ.
  - Joint nào CTR-GCN cho trọng số cao → cột đó sáng lên (rõ hơn).
  - Joint nào trọng số thấp → cột đó mờ đi.
  
  Ví dụ: Tay có 4 joints (Shoulder, Elbow, Wrist, Hand).
         Nếu CTR-GCN nói Wrist quan trọng nhất → vùng cột Wrist sáng,
         Shoulder ít quan trọng → vùng cột Shoulder tối hơn.

Flow:
  1. Load ảnh skeleton-to-image đã tạo bởi skeleton_to_image.py
  2. Load skeleton → feed CTR-GCN → extract per-joint importance
  3. Map 20 joint weights theo UCLA_JOINT_REORDER (giống skeleton_to_image.py)
  4. Tạo attention map: 1 cột = 1 weight → resize lên (224, 224)
  5. Nhân attention lên ảnh → clip → save

Cách dùng:
  python gen_skeleton_image_weighted.py \
      --weights ./result/nucla/CTROGC-GCN.pt \
      --input ./skeleton_images \
      --output ./skeleton_images_weighted
"""

import os
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.getcwd())

from models.ctrgcn import Model as CTRGCN
from feeder.feeder_nucla_fusion import Feeder

# ============================================================================
# Config
# ============================================================================
OUTPUT_SIZE = 224

output_device = 0 if torch.cuda.is_available() else 'cpu'
device = torch.device(f"cuda:{output_device}" if torch.cuda.is_available() else "cpu")

# Joint reorder — PHẢI GIỐNG HỆT skeleton_to_image.py
UCLA_JOINT_REORDER = [
    3, 2, 1, 0,         # Group 1 - Spine
    4, 5, 6, 7,          # Group 2 - Left Arm
    8, 9, 10, 11,        # Group 3 - Right Arm
    12, 13, 14, 15,      # Group 4 - Left Leg
    16, 17, 18, 19,      # Group 5 - Right Leg
]

JOINT_NAMES = [
    'Head', 'ShldrCtr', 'Spine', 'HipCtr',           # Spine group
    'ShldrL', 'ElbowL', 'WristL', 'HandL',           # Left Arm
    'ShldrR', 'ElbowR', 'WristR', 'HandR',           # Right Arm
    'HipL', 'KneeL', 'AnkleL', 'FootL',              # Left Leg
    'HipR', 'KneeR', 'AnkleR', 'FootR',              # Right Leg
]

NUM_JOINTS = len(UCLA_JOINT_REORDER)  # 20


def extract_per_joint_importance(model, skeleton_tensor):
    """
    Extract per-joint importance score từ CTR-GCN.
    
    Args:
        model: CTR-GCN model (eval mode)
        skeleton_tensor: (1, C, T, V, M) tensor on device
    
    Returns:
        joint_importance: numpy array shape (V,) — importance per joint (original order)
    """
    N, C, T, V, M = skeleton_tensor.size()
    
    with torch.no_grad():
        _, feature = model.extract_feature(skeleton_tensor)
        # feature shape: (N, C_feat, T_feat, V, M)
    
    # L2 norm across feature channels → per-joint-per-timestep magnitude
    # (N, C_feat, T_feat, V, M) → sum over C_feat → (N, T_feat, V, M)
    intensity = (feature * feature).sum(dim=1) ** 0.5
    intensity = intensity.cpu().detach().numpy()  # (N, T_feat, V, M)
    
    # Chọn person chính (nếu M > 1)
    person_idx = 0
    if M > 1:
        if intensity[0, :, :, 0].mean() < intensity[0, :, :, 1].mean():
            person_idx = 1
    
    # Per-joint importance: trung bình theo temporal axis
    # intensity[0, :, :, person_idx] shape: (T_feat, V)
    joint_importance = intensity[0, :, :, person_idx].mean(axis=0)  # (V,)
    
    return joint_importance


def create_per_joint_attention_map(joint_weights_reordered, output_size=OUTPUT_SIZE):
    """
    Tạo 2D attention map từ per-joint weights.
    
    Trong skeleton-to-image:
      - Width (trục ngang) = joints (sau reorder) 
      - Height (trục dọc) = temporal frames
    
    → Mỗi cột (joint) có weight riêng.
    → Resize từ (1, num_joints) lên (output_size, output_size).
    
    Args:
        joint_weights_reordered: (num_joints,) weights đã theo thứ tự reorder
        output_size: kích thước ảnh
    
    Returns:
        attention_map: (output_size, output_size) float32
    """
    # 1D weight vector → (1, num_joints) 
    attn_1d = joint_weights_reordered.reshape(1, -1).astype(np.float32)
    
    # Resize lên (output_size, output_size)
    # Mỗi "cột" joint sẽ được kéo dãn ra theo cả 2 chiều
    attn_map = cv2.resize(attn_1d, (output_size, output_size), 
                          interpolation=cv2.INTER_LINEAR)
    
    return attn_map


def generate_weighted_skeleton_images(
    weights_path, input_path, output_path, output_size=OUTPUT_SIZE
):
    """
    Apply per-joint soft attention lên ảnh skeleton-to-image.
    
    Args:
        weights_path: path tới file CTR-GCN weights (.pt)
        input_path: path tới folder chứa ảnh skeleton-to-image (output của skeleton_to_image.py)
        output_path: path lưu ảnh weighted
        output_size: kích thước ảnh
    """
    os.makedirs(output_path, exist_ok=True)

    # --- Khởi tạo CTR-GCN ---
    print(">>> Đang khởi tạo CTR-GCN...")
    model = CTRGCN(
        num_class=10,
        num_point=20,
        num_person=1,
        graph='graph.ucla.Graph',
        graph_args={'labeling_mode': 'spatial'},
        in_channels=3,
        drop_out=0,
        adaptive=True
    ).to(device)

    if weights_path and os.path.exists(weights_path):
        print(f">>> Loading weights: {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(">>> Load weights thành công!")
        except Exception as e:
            print(f"!!! Cảnh báo: Không load được ({e}). Dùng weights ngẫu nhiên.")
    else:
        print(f"!!! Không tìm thấy: {weights_path}")

    model.eval()

    # --- Process từng split ---
    splits = ['train', 'val']
    total_count = 0
    skip_count = 0
    debug_count = 0
    DEBUG_LIMIT = 5

    for split in splits:
        print(f"\n>>> Đang xử lý split: {split}")
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

        for i, (data, label, index) in enumerate(tqdm(loader, desc=f"  {split}")):
            # data[0] = skeleton: (N, C, T, V, M)
            data_ske = data[0].float().to(device)

            idx = index.item()
            file_name = feeder.data_dict[idx]['file_name']

            # --- Load ảnh skeleton-to-image gốc ---
            img_path = os.path.join(input_path, f"{file_name}.png")
            if not os.path.exists(img_path):
                skip_count += 1
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                skip_count += 1
                continue
            img_float = img_bgr.astype(np.float32)  # (H, W, 3) BGR, [0, 255]

            # --- Extract per-joint importance từ CTR-GCN ---
            joint_importance = extract_per_joint_importance(model, data_ske)
            # joint_importance shape: (20,) — original joint order

            # Reorder theo UCLA_JOINT_REORDER (giống skeleton_to_image.py)
            joint_importance_reordered = joint_importance[UCLA_JOINT_REORDER]

            # Normalize về [0.5, 1.5]
            # → Joint quan trọng nhất: sáng lên 50% (×1.5)
            # → Joint ít quan trọng nhất: mờ đi 50% (×0.5)
            w_min, w_max = joint_importance_reordered.min(), joint_importance_reordered.max()
            if (w_max - w_min) > 1e-8:
                joint_weights = 0.5 + 1.0 * (joint_importance_reordered - w_min) / (w_max - w_min)
            else:
                joint_weights = np.ones(NUM_JOINTS)

            # Debug: in weights cho vài sample đầu
            if debug_count < DEBUG_LIMIT:
                print(f"\n  [DEBUG] {file_name}:")
                for j in range(NUM_JOINTS):
                    bar = '█' * int(joint_weights[j] * 20)
                    print(f"    {JOINT_NAMES[j]:>10s} [{j:2d}]: {joint_weights[j]:.3f} {bar}")
                debug_count += 1

            # --- Tạo attention map và apply ---
            attn_map = create_per_joint_attention_map(joint_weights, output_size)
            # attn_map shape: (H, W), mỗi cột = weight của 1 joint

            # Nhân attention lên cả 3 channels
            weighted_img = img_float * attn_map[:, :, np.newaxis]

            # Clip + uint8
            weighted_img = np.clip(weighted_img, 0, 255).astype(np.uint8)

            # Save
            save_path = os.path.join(output_path, f"{file_name}.png")
            cv2.imwrite(save_path, weighted_img)
            total_count += 1

    print(f"\n{'='*60}")
    print(f"Hoàn tất! Đã tạo {total_count} ảnh weighted tại: {output_path}")
    if skip_count > 0:
        print(f"Bỏ qua {skip_count} mẫu (không tìm thấy ảnh gốc)")


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply per-joint soft attention (CTR-GCN) lên skeleton-to-image"
    )
    parser.add_argument(
        '--weights', '-w', type=str,
        default='./result/nucla/CTROGC-GCN.pt',
        help='Path tới CTR-GCN weights (.pt)'
    )
    parser.add_argument(
        '--input', '-i', type=str,
        default='./skeleton_images',
        help='Path tới folder ảnh skeleton-to-image (output của skeleton_to_image.py)'
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

    generate_weighted_skeleton_images(
        weights_path=args.weights,
        input_path=args.input,
        output_path=args.output,
        output_size=args.size,
    )
