import os
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ctrgcn import Model as CTRGCNModel
from feeder.feeder_nucla_gcn import Feeder

RGB_PATH = 'C:/ucla_stroi/'
OUTPUT_PATH = 'C:/ucla_stroi_weighted/'
DATA_PATH = 'C:/Users/nguyn/Downloads/NW-UCLA-ALL/NW-UCLA-ALL'
WEIGHTS_PATH = './result/nucla/CTROGC-GCN.pt'

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    print("Loading CTR-GCN model...")
    # Initialize the teacher CTR-GCN
    model = CTRGCNModel(num_class=10, num_point=20, num_person=1, graph="graph.ucla.Graph", graph_args={'labeling_mode': 'spatial'})
    
    weights = torch.load(WEIGHTS_PATH)
    if 'model' in weights:
        model.load_state_dict(weights['model'])
    else:
        model.load_state_dict(weights)
        
    model.cuda()
    model.eval()

    print("Iterating over dataset to generate weighted STROI images...")
    processed = 0
    missing = 0
    for split in ['train', 'val']:
        # We set random augmentations to False so that x_ gives absolute spatial inputs identical across runs.
        feeder = Feeder(data_path=DATA_PATH, label_path=split, random_choose=False, random_shift=False, random_move=False, window_size=52, normalization=False, debug=False, use_mmap=True)
        
        for i in tqdm(range(len(feeder)), desc=f"Processing {split} split"):
            data, _, label, index = feeder[i]
            sample_info = feeder.data_dict[index]
            file_name = sample_info['file_name']
            
            in_img_path = os.path.join(RGB_PATH, file_name + '.png')
            if not os.path.exists(in_img_path):
                missing += 1
                continue
                
            x_ = torch.from_numpy(data).unsqueeze(0).cuda() # (1, C, T, V, M)
            
            with torch.no_grad():
                predict, feature = model.extract_feature(x_)
                intensity_s = (feature*feature).sum(dim=1)**0.5 # (N, T, V, M)
                intensity_s = intensity_s.cpu().detach().numpy()
                feature_s = np.abs(intensity_s)
                
                f_min = feature_s.min()
                f_max = feature_s.max()
                if f_max > f_min:
                    feature_s = 255 * (feature_s - f_min) / (f_max - f_min)
                else:
                    feature_s = np.zeros_like(feature_s)

                temporal_positions = 15
                parts_v = [3, 11, 7, 18, 14]
                
                try:
                    rgb_img = Image.open(in_img_path).convert('RGB')
                except:
                    missing += 1
                    continue
                
                img_np = np.array(rgb_img).astype(np.float32)
                H, W, C = img_np.shape
                PART_SIZE = H // 5 # Standard STROI has 5 stacked parts vertically
                
                for j, v in enumerate(parts_v):
                    part_feat = feature_s[0, :, v, 0] # shape (T,)
                    # Get top 'temporal_positions' values using partition along time dim
                    if len(part_feat) >= temporal_positions:
                        temp = np.partition(-part_feat, temporal_positions)
                        part_val = -temp[:temporal_positions].mean()
                    else:
                        part_val = part_feat.mean()
                        
                    weight_scalar = part_val / 127.0
                    
                    # Apply computed specific structural body part weight
                    img_np[j*PART_SIZE:(j+1)*PART_SIZE, :, :] *= weight_scalar
                    
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                weighted_img = Image.fromarray(img_np)
                
                out_img_path = os.path.join(OUTPUT_PATH, file_name + '.png')
                weighted_img.save(out_img_path)
                processed += 1
                
    print(f"Done! Created {processed} weighted images in {OUTPUT_PATH}. Missing {missing} base images.")

if __name__ == '__main__':
    main()
