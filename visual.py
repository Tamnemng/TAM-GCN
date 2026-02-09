import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2
sys.path.append(os.getcwd())
from models.ctrgcn import Model as CTRGCN
from feeder.feeder_nucla_fusion import Feeder
output_device = 0 if torch.cuda.is_available() else 'cpu'
device = torch.device(f"cuda:{output_device}" if torch.cuda.is_available() else "cpu")

def visualize_fusion_effect(dataset_path, weights_path=None):
    print(">>> Đang khởi tạo Feeder...")
    feeder = Feeder(
        split='train', 
        random_choose=False, 
        random_shift=False, 
        random_move=False,
        window_size=50, # Window size mặc định cho UCLA
        temporal_rgb_frames=5
    )
    data, label, index = feeder[0]
    data_ske = data[0]
    data_rgb = data[1]
    x_ske = torch.tensor(data_ske).unsqueeze(0).float().to(device)
    x_rgb = torch.tensor(data_rgb).unsqueeze(0).float().to(device) 

    print(f">>> Đã load mẫu dữ liệu. Shape Xương: {x_ske.shape}, Shape Ảnh: {x_rgb.shape}")

    print(">>> Đang khởi tạo mô hình CTR-GCN...")
    graph_args = {'labeling_mode': 'spatial'}
    model_ske = CTRGCN(
        num_class=10, # UCLA có 10 class
        num_point=20, 
        num_person=1,
        graph='graph.ucla.Graph',
        graph_args=graph_args,
        in_channels=3
    ).to(device)
    if weights_path and os.path.exists(weights_path):
        print(f">>> Loading weights từ: {weights_path}")
        try:
            model_ske.load_state_dict(torch.load(weights_path))
        except Exception as e:
            print(f"!!! Cảnh báo: Không load được weights ({e}). Dùng weights ngẫu nhiên.")
    else:
        print("!!! Cảnh báo: Không tìm thấy weights path. Đang chạy với weights ngẫu nhiên (kết quả heatmap có thể sẽ nhiễu).")

    model_ske.eval()
    print(">>> Đang tính toán Attention Map...")
    with torch.no_grad():
        _, feature = model_ske.extract_feature(x_ske)
    intensity_s = (feature*feature).sum(dim=1)**0.5
    intensity_s = intensity_s.cpu().detach().numpy()
    feature_s = np.abs(intensity_s)
    if (feature_s.max() - feature_s.min()) != 0:
        feature_s = 255 * (feature_s-feature_s.min()) / (feature_s.max()-feature_s.min())
    
    N, C, T, V, M = x_ske.size()
    temporal_positions = 15
    temporal_rgb_frames = 5
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
                temp = np.partition(-feature_val, min(temporal_positions, len(feature_val)-1))
                avg_feat = -temp[:temporal_positions].mean()
                if 45*(j+1) <= target_w_map:
                    weight[n, 0, 45*j:45*(j+1), :] = avg_feat
    weight_cuda = torch.from_numpy(weight).float().to(device)
    weight_cuda = weight_cuda / 127.0 
    _, _, H_rgb, W_rgb = x_rgb.size()
    weight_resized = torch.nn.functional.interpolate(weight_cuda, size=(H_rgb, W_rgb), mode='bilinear', align_corners=False)
    rgb_weighted = x_rgb * weight_resized
    img_origin = x_rgb[0, :3, :, :].permute(1, 2, 0).cpu().numpy()
    img_weight = weight_resized[0, 0, :, :].cpu().numpy()
    img_fusion = rgb_weighted[0, :3, :, :].permute(1, 2, 0).cpu().numpy()
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
    WEIGHTS_PATH = './result/nucla/CTROGC-GCN.pt' 
    
    visualize_fusion_effect('./data/nucla/', WEIGHTS_PATH)