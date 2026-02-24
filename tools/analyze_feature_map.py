import os
import sys
import numpy as np
import json
import torch
from collections import Counter

sys.path.append('./')
from models.ctrgcn import Model as CTRGCNModel
from tools.gen_ucla_stroi_topology import load_3d_skeleton, BODY_PARTS, PART_NAMES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_feature_map():
    JSON_DIR = r'C:\Users\nguyn\Downloads\NW-UCLA-ALL\NW-UCLA-ALL'
    
    label_samples = {i: [] for i in range(1, 11)}
    
    for f in os.listdir(JSON_DIR):
        if os.path.isdir(os.path.join(JSON_DIR, f)):
            parts = f.split('_')
            try:
                action_id = int(parts[0][1:])
                if action_id in label_samples:
                    label_samples[action_id].append(f)
            except:
                pass
                
    model = CTRGCNModel(num_class=10, num_point=20, num_person=1, graph='graph.ucla.Graph', graph_args={'labeling_mode': 'spatial'}, in_channels=3)
    weights = torch.load('./result/nucla/CTROGC-GCN.pt', map_location=device)
    model.load_state_dict(weights)
    model.eval()
    model = model.to(device)
    
    results_data = {}
    
    for label in range(1, 11):
        samples = label_samples[label][:20]
        if not samples: continue
        
        top_parts = []
        
        for sample in samples:
            x_3d = load_3d_skeleton(sample, JSON_DIR)
            if x_3d is None: continue
            
            x_tensor = torch.tensor(x_3d).unsqueeze(0).float().to(device)
            with torch.no_grad():
                feat, _ = model.extract_feature(x_tensor) # [N, C, T, V, M]
            
            # Tính trung bình cường độ trên Channel, Frame, M để lấy tầm quan trọng của 20 Khớp
            feat_abs = feat.abs()
            weight_V = feat_abs.mean(dim=(1, 2, 4)).squeeze() # shape [20]
            weight_V = weight_V.cpu().numpy()
            
            part_weights = []
            for p_name in PART_NAMES:
                idx = BODY_PARTS[p_name]
                part_weights.append(np.sum(weight_V[idx]))
                
            top_part_idx = np.argmax(part_weights)
            top_parts.append(PART_NAMES[top_part_idx])
            
        counts = Counter(top_parts)
        top_part, count = counts.most_common(1)[0] if counts else ("None", 0)
        total = len(top_parts)
        
        results_data[f"a{label:02d}"] = {
            "top_part": top_part,
            "consistency": f"{(count/total)*100:.1f}%",
            "distribution": dict(counts)
        }

    with open('feature_map_analysis.json', 'w') as f:
        json.dump(results_data, f, indent=4)
        
    print("DONE! Saved to feature_map_analysis.json")

if __name__ == '__main__':
    analyze_feature_map()
