import torch
import torch.nn as nn
import numpy as np
import sys
import os
import cv2
from tqdm import tqdm
from PIL import Image

sys.path.append(os.getcwd())
from models.ctrgcn import Model as CTRGCN
from feeder.feeder_nucla_fusion import Feeder

output_device = 0 if torch.cuda.is_available() else 'cpu'
device = torch.device(f"cuda:{output_device}" if torch.cuda.is_available() else "cpu")

def generate_weighted_images(weights_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(">>> Đang khởi tạo mô hình CTR-GCN...")
    graph_args = {'labeling_mode': 'spatial'}
    model_ske = CTRGCN(
        num_class=10, 
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
        print("!!! Cảnh báo: Không tìm thấy weights path. Đang chạy với weights ngẫu nhiên.")

    model_ske.eval()

    splits = ['train', 'val']
    
    for split in splits:
        print(f">>> Đang xử lý tập dữ liệu: {split}")
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

        for i, (data, label, index) in enumerate(tqdm(loader)):
            # Get data
            data_ske = data[0].float().to(device)
            data_rgb = data[1].float().to(device)
            
            # Get filename using index
            idx = index.item()
            file_name = feeder.data_dict[idx]['file_name']
            
            # Extract features
            with torch.no_grad():
                _, feature = model_ske.extract_feature(data_ske)
                
            intensity_s = (feature*feature).sum(dim=1)**0.5
            intensity_s = intensity_s.cpu().detach().numpy()
            feature_s = np.abs(intensity_s)
            
            if (feature_s.max() - feature_s.min()) != 0:
                feature_s = 255 * (feature_s-feature_s.min()) / (feature_s.max()-feature_s.min())
            
            N, C, T, V, M = data_ske.size()
            
            # Parameters for weight map
            temporal_positions = 15
            temporal_rgb_frames = 5
            target_h_map = 225
            target_w_map = 45 * temporal_rgb_frames
            
            weight = np.full((N, 1, target_h_map, target_w_map), 0.0) 
            target_joints = [3, 11, 7, 18, 14] 
            
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
            
            _, _, H_rgb, W_rgb = data_rgb.size()
            weight_resized = torch.nn.functional.interpolate(weight_cuda, size=(H_rgb, W_rgb), mode='bilinear', align_corners=False)
            
            rgb_weighted = data_rgb * weight_resized
            
            # Save format: assuming visual.py saves the first 3 channels equivalent (RGB)
            # If data_rgb has more channels (stacked frames), we might only be saving one "view".
            # Original visual.py: img_fusion = rgb_weighted[0, :3, :, :]....
            
            # Use same logic as visual.py to get the output image
            img_fusion = rgb_weighted[0, :3, :, :].permute(1, 2, 0).cpu().numpy()
            img_fusion = (img_fusion - img_fusion.min()) / (img_fusion.max() - img_fusion.min())
            img_fusion = (img_fusion * 255).astype(np.uint8)
            
            # Save
            save_name = f"{file_name}.png"
            save_full_path = os.path.join(output_path, save_name)
            Image.fromarray(img_fusion).save(save_full_path)

    print(f"\nHoàn tất! Ảnh được lưu tại: {output_path}")

if __name__ == '__main__':
    WEIGHTS_PATH = './result/nucla/CTROGC-GCN.pt' 
    OUTPUT_PATH = './ucla_stroi_weighted/'
    
    generate_weighted_images(WEIGHTS_PATH, OUTPUT_PATH)
