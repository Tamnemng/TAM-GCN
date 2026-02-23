import os
import sys
import numpy as np
import json
import torch
import types
import argparse
from PIL import Image, ImageDraw
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
            return None
        
        if len(skel_data.shape) == 2:
            T, _ = skel_data.shape
            skel_data = skel_data.reshape(T, 20, 3)
            
        # (T, V, C) -> (C, T, V)
        data_numpy = skel_data.transpose(2, 0, 1) 
        # (C, T, V, M)
        data_numpy = np.expand_dims(data_numpy, axis=-1) 
        
        # padding/trimming to exactly 52 frames cho GCN
        C, T, V, M = data_numpy.shape
        window_size = 52
        if T < window_size:
            pad = np.zeros((C, window_size - T, V, M))
            data_numpy = np.concatenate((data_numpy, pad), axis=1)
        elif T > window_size:
            begin = (T - window_size) // 2
            data_numpy = data_numpy[:, begin:begin+window_size, :, :]
            
        return data_numpy
    except Exception as e:
        # print(f"Lỗi load skeleton {sample_name}: {e}")
        return None

def hooked_forward(self, x, A=None, alpha=1):
    x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
    x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
    # Sinh ra Ma trận kề Động của GCN mang đặc trưng quan hệ topology
    A_dyn = self.conv4(x1) * alpha
    # Cất giấu bản Động (Dynamic) đi để sau đó mình lôi ra dùng
    self.saved_A_dyn = A_dyn.detach()
    A_full = A_dyn + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
    out = torch.einsum('ncuv,nctv->nctu', A_full, x3)
    return out

def extract_related_parts(model, x_3d):
    x_tensor = torch.tensor(x_3d).unsqueeze(0).float().to(device)
    with torch.no_grad():
        model.extract_feature(x_tensor)
        
    # Trích topology động từ layer 1 (hoặc thay bằng model.l10 tuỳ ý)
    layer = getattr(model, 'l1')
    A_total = []
    
    for i in range(layer.gcn1.num_subset):
        A_dyn = layer.gcn1.convs[i].saved_A_dyn # [1, C, V, V]
        A_avg = A_dyn.abs().mean(dim=(0, 1)) # Tính trung bình magnitude trên Channel -> [V, V]
        A_total.append(A_avg)
    A_total = torch.stack(A_total).mean(dim=0)
    
    # Cộng dồn 2 chiều để ra ma trận đối xứng vô hướng giữa 2 Khớp u và v
    A_sym = (A_total + A_total.t()) / 2.0
    A_sym = A_sym.cpu().numpy()
    
    # Gom 20 khớp thành ma trận 5x5 đại diện cho 5 bộ phận
    part_matrix = np.zeros((5, 5))
    for i, p1 in enumerate(PART_NAMES):
        for j, p2 in enumerate(PART_NAMES):
            idx1 = BODY_PARTS[p1]
            idx2 = BODY_PARTS[p2]
            sub_A = A_sym[np.ix_(idx1, idx2)]
            part_matrix[i, j] = np.sum(sub_A)
            
    np.fill_diagonal(part_matrix, -np.inf)
    
    # Bộ phận nào liên kết mạnh nhất?
    flat_idx = np.argmax(part_matrix)
    c1, c2 = divmod(flat_idx, 5)
    return c1, c2

def highlight_rows(image, r1, r2, color=(200, 200, 0, 75)): 
    # color: dải RGBA. (Màu Vàng Đồng/Hơi mờ, có thể đổi tùy ý)
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    width = image.width
    
    # Nhuộm băng r1
    y1_start = r1 * PART_SIZE
    y1_end = y1_start + PART_SIZE
    draw.rectangle([0, y1_start, width, y1_end], fill=color)
    
    # Nhuộm băng r2
    if r1 != r2:
        y2_start = r2 * PART_SIZE
        y2_end = y2_start + PART_SIZE
        draw.rectangle([0, y2_start, width, y2_end], fill=color)
        
    # Chồng lại với ảnh thật
    image = image.convert('RGBA')
    image = Image.alpha_composite(image, overlay)
    return image.convert('RGB')

def process_all(input_dir, output_dir, json_data_dir):
    if not os.path.exists(input_dir):
        print(f"LỖI: Thư mục chứa ảnh Stroi '{input_dir}' không tồn tại!")
        print("Hãy chạy gen_ucla_stroi.py trước hoặc cung cấp đúng đường dẫn với biến --input_dir")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nKhởi tạo mô hình CTR-GCN để trích xuất topology...")
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
    
    # Hack/Patch hàm chạy của CTR-GCN để hứng lấy Topology
    for i in range(model.l1.gcn1.num_subset):
        model.l1.gcn1.convs[i].forward = types.MethodType(hooked_forward, model.l1.gcn1.convs[i])

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png') or f.endswith('.jpg')]
    if len(image_files) == 0:
        print(f"Không tìm thấy ảnh .png nào trong thư mục {input_dir}")
        return
        
    print(f"\nBắt đầu lướt qua {len(image_files)} ảnh 5x5 đã tạo và nhuộm màu Topology...")
    
    count = 0
    tbar = tqdm(image_files)
    for img_file in tbar:
        sample_name = os.path.splitext(img_file)[0]
        
        # 1. Load ảnh Stroi gốc
        img_path = os.path.join(input_dir, img_file)
        try:
            img = Image.open(img_path)
            # Kiểm tra ảnh hợp lệ 480x480 (5 bộ phận * 96) không?
            if img.height < 5 * PART_SIZE:
                continue
        except Exception as e:
            # print(f"Không mở được ảnh {img_path}: {e}")
            continue
            
        # 2. Xử lý Skeleton vào GCN
        x_3d = load_3d_skeleton(sample_name, json_data_dir)
        if x_3d is not None:
            c1, c2 = extract_related_parts(model, x_3d)
            tbar.set_description(f"Cặp ({c1}, {c2}) - {sample_name}")
            
            # 3. Nhuộm Overlay lên ảnh gốc và đi lưu
            # Tô viền mờ băng ghi hình của bộ phận (c1) và (c2)
            highlighted_img = highlight_rows(img, c1, c2, color=(200, 200, 0, 75)) # Màu Vàng Nhạt
            
            save_path = os.path.join(output_dir, img_file)
            highlighted_img.save(save_path)
            count += 1
        # else:
        #     print(f"Bỏ qua {sample_name} vì thiếu file .json")
            
    print(f"\nHoàn tất! Đã lưu {count} ảnh 5x5 có nhuộm màu Topology Map vào:\n ---> {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Đắp màu Topology lên ảnh Stroi")
    # Thay đổi mặc định dựa theo logic code cũ gen_ucla_stroi.py (của bản thân user) 
    parser.add_argument('--input_dir', type=str, default='/ucla_stroi', help='Thư mục chứa ảnh 5x5 Stroi đầu vào')
    parser.add_argument('--output_dir', type=str, default='./ucla_stroi_topology', help='Thư mục chứa ảnh đã tô màu')
    parser.add_argument('--JSON_DATA_DIR', type=str, default=r'C:\Users\nguyn\Downloads\NW-UCLA-ALL\NW-UCLA-ALL', help='Thư mục chứa file JSON 3D Skeletons của UCLA')
    args = parser.parse_args()
    
    process_all(args.input_dir, args.output_dir, args.JSON_DATA_DIR)
