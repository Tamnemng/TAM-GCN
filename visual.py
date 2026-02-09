import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2

# Thêm đường dẫn hiện tại vào sys.path để import được các module trong project
sys.path.append(os.getcwd())

# Import các module cần thiết từ code của bạn
from models.ctrgcn import Model as CTRGCN
from feeder.feeder_nucla_fusion import Feeder

# Cấu hình thiết bị
output_device = 0 if torch.cuda.is_available() else 'cpu'
device = torch.device(f"cuda:{output_device}" if torch.cuda.is_available() else "cpu")

def visualize_fusion_effect(dataset_path, weights_path=None):
    print(">>> Đang khởi tạo Feeder...")
    # Cấu hình Feeder (Lấy 1 mẫu từ tập train hoặc test)
    # Bạn có thể cần chỉnh lại 'data_path' hoặc 'split' cho đúng với đường dẫn máy bạn
    feeder = Feeder(
        split='train', 
        random_choose=False, 
        random_shift=False, 
        random_move=False,
        window_size=50, # Window size mặc định cho UCLA
        temporal_rgb_frames=5
    )
    
    # Lấy mẫu đầu tiên (index 0)
    data, label, index = feeder[0]
    data_ske = data[0] # Shape: (3, T, V, M)
    data_rgb = data[1] # Shape: (C, H, W) - RGB images (ST-ROI)

    # Chuyển sang Tensor và đưa lên GPU
    x_ske = torch.tensor(data_ske).unsqueeze(0).float().to(device) # Thêm batch dim -> (1, 3, T, V, M)
    x_rgb = torch.tensor(data_rgb).unsqueeze(0).float().to(device) # Thêm batch dim -> (1, C, H, W)

    print(f">>> Đã load mẫu dữ liệu. Shape Xương: {x_ske.shape}, Shape Ảnh: {x_rgb.shape}")

    print(">>> Đang khởi tạo mô hình CTR-GCN...")
    # Cấu hình Graph cho UCLA (20 joints)
    graph_args = {'labeling_mode': 'spatial'}
    
    # Khởi tạo model CTR-GCN (chỉ phần xương)
    model_ske = CTRGCN(
        num_class=10, # UCLA có 10 class
        num_point=20, 
        num_person=1,
        graph='graph.ucla.Graph',
        graph_args=graph_args,
        in_channels=3
    ).to(device)

    # Nếu có file weights đã train, load vào đây để visualization chính xác hơn
    if weights_path and os.path.exists(weights_path):
        print(f">>> Loading weights từ: {weights_path}")
        try:
            model_ske.load_state_dict(torch.load(weights_path))
        except Exception as e:
            print(f"!!! Cảnh báo: Không load được weights ({e}). Dùng weights ngẫu nhiên.")
    else:
        print("!!! Cảnh báo: Không tìm thấy weights path. Đang chạy với weights ngẫu nhiên (kết quả heatmap có thể sẽ nhiễu).")

    model_ske.eval()

    # --- BẮT ĐẦU LOGIC FUSION (Copy từ models/mmn_ctrgcn.py) ---
    print(">>> Đang tính toán Attention Map...")
    with torch.no_grad():
        _, feature = model_ske.extract_feature(x_ske)

    # Tính intensity từ feature xương
    intensity_s = (feature*feature).sum(dim=1)**0.5
    intensity_s = intensity_s.cpu().detach().numpy()
    
    # Chuẩn hóa về 0-255
    feature_s = np.abs(intensity_s)
    if (feature_s.max() - feature_s.min()) != 0:
        feature_s = 255 * (feature_s-feature_s.min()) / (feature_s.max()-feature_s.min())
    
    N, C, T, V, M = x_ske.size()
    
    # Các tham số fix cứng trong file models/mmn_ctrgcn.py của bạn
    temporal_positions = 15
    temporal_rgb_frames = 5
    
    # Lưu ý: Code gốc của bạn định nghĩa weight size là (225, 45*frames)
    # Nhưng ảnh RGB load từ feeder thường là 224x224. Tôi sẽ giữ logic tạo weight
    # và resize nó cho khớp với ảnh thật để hiển thị.
    target_h_map = 225
    target_w_map = 45 * temporal_rgb_frames
    
    weight = np.full((N, 1, target_h_map, target_w_map), 0.0) 
    target_joints = [3, 11, 7, 18, 14] # 5 khớp mục tiêu
    
    for n in range(N):
        person_idx = 0
        if M > 1:
            if feature_s[n, :, :, 0].mean() < feature_s[n, :, :, 1].mean():
                person_idx = 1
        
        for j, v in enumerate(target_joints):
            if v < V:
                feature_val = feature_s[n, :, v, person_idx]
                # Lấy giá trị trung bình của top features
                temp = np.partition(-feature_val, min(temporal_positions, len(feature_val)-1))
                avg_feat = -temp[:temporal_positions].mean()
                
                # Điền vào vùng tương ứng trên bản đồ trọng số
                # Mỗi khớp chiếm 1 cột dọc rộng 45 pixel
                if 45*(j+1) <= target_w_map:
                    weight[n, 0, 45*j:45*(j+1), :] = avg_feat

    # Chuyển weight sang Tensor và normalize
    weight_cuda = torch.from_numpy(weight).float().to(device)
    weight_cuda = weight_cuda / 127.0 
    
    # --- XỬ LÝ ĐỂ HIỂN THỊ ---
    
    # Lấy kích thước thật của ảnh RGB input
    _, _, H_rgb, W_rgb = x_rgb.size()
    
    # Resize weight map cho khớp với kích thước ảnh RGB thật (ví dụ 224x224)
    weight_resized = torch.nn.functional.interpolate(weight_cuda, size=(H_rgb, W_rgb), mode='bilinear', align_corners=False)
    
    # Áp dụng trọng số: Ảnh gốc * Trọng số
    rgb_weighted = x_rgb * weight_resized

    # Convert về numpy để vẽ (Lấy sample đầu tiên, frame RGB đầu tiên)
    # x_rgb shape: (1, 3*frames, H, W). Lấy 3 kênh đầu tiên làm ảnh đại diện
    img_origin = x_rgb[0, :3, :, :].permute(1, 2, 0).cpu().numpy()
    img_weight = weight_resized[0, 0, :, :].cpu().numpy()
    img_fusion = rgb_weighted[0, :3, :, :].permute(1, 2, 0).cpu().numpy()

    # Chuẩn hóa lại ảnh RGB để hiển thị (về khoảng 0-1)
    img_origin = (img_origin - img_origin.min()) / (img_origin.max() - img_origin.min())
    img_fusion = (img_fusion - img_fusion.min()) / (img_fusion.max() - img_fusion.min())

    # Vẽ hình
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_origin)
    plt.title("Ảnh gốc (ST-ROI)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_weight, cmap='jet')
    plt.title("Weight Map (Từ Skeleton)")
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_fusion)
    plt.title("Kết quả (Ảnh gốc * Weight)")
    plt.axis('off')

    plt.tight_layout()
    output_file = 'visualization_result.png'
    plt.savefig(output_file)
    print(f">>> Đã lưu ảnh kết quả vào: {output_file}")
    plt.show()

if __name__ == '__main__':
    # ĐƯỜNG DẪN WEIGHTS (Quan trọng)
    # Bạn hãy thay thế đường dẫn này bằng đường dẫn file .pt của CTR-GCN đã train
    # Ví dụ: './work_dir/nucla/ctrgcn/runs-50.pt'
    # Nếu không có, để None (sẽ dùng weight ngẫu nhiên -> heatmap sẽ trông như nhiễu TV)
    WEIGHTS_PATH = './result/nucla/CTROGC-GCN.pt' 
    
    visualize_fusion_effect('./data/nucla/', WEIGHTS_PATH)