import os
import sys
import numpy as np
import json
import torch
import types
from collections import Counter

sys.path.append('./')
from models.ctrgcn import Model as CTRGCNModel
from tools.gen_ucla_stroi_topology import load_3d_skeleton, BODY_PARTS, PART_NAMES, hooked_forward

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_layers():
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
    
    for l_idx in range(1, 11):
        layer = getattr(model, f'l{l_idx}')
        for i in range(layer.gcn1.num_subset):
            layer.gcn1.convs[i].forward = types.MethodType(hooked_forward, layer.gcn1.convs[i])

    results_data = {}
    
    for label in range(1, 11):
        samples = label_samples[label][:20]
        if not samples: continue
        
        layer_results = {l_idx: [] for l_idx in range(1, 11)}
        
        for sample in samples:
            x_3d = load_3d_skeleton(sample, JSON_DIR)
            if x_3d is None: continue
            
            x_tensor = torch.tensor(x_3d).unsqueeze(0).float().to(device)
            with torch.no_grad():
                model.extract_feature(x_tensor)
                
            for l_idx in range(1, 11):
                layer = getattr(model, f'l{l_idx}')
                A_total = []
                for i in range(layer.gcn1.num_subset):
                    A_dyn = layer.gcn1.convs[i].saved_A_dyn
                    A_avg = A_dyn.abs().mean(dim=(0, 1))
                    A_total.append(A_avg)
                A_total = torch.stack(A_total).mean(dim=0)
                A_sym = (A_total + A_total.t()) / 2.0
                A_sym = A_sym.cpu().numpy()
                
                part_matrix = np.zeros((5, 5))
                for i, p1 in enumerate(PART_NAMES):
                    for j, p2 in enumerate(PART_NAMES):
                        idx1 = BODY_PARTS[p1]
                        idx2 = BODY_PARTS[p2]
                        sub_A = A_sym[np.ix_(idx1, idx2)]
                        part_matrix[i, j] = np.sum(sub_A)
                
                np.fill_diagonal(part_matrix, -np.inf)
                flat_idx = np.argmax(part_matrix)
                c1, c2 = divmod(flat_idx, 5)
                
                # Sắp xếp alphabet để đảm bảo tính duy nhất của cặp
                pairs = sorted([PART_NAMES[c1], PART_NAMES[c2]])
                pair_name = f"{pairs[0]}-{pairs[1]}"
                layer_results[l_idx].append(pair_name)
                
        results_data[f"a{label:02d}"] = {}
        for l_idx in range(1, 11):
            counts = Counter(layer_results[l_idx])
            top_pair, count = counts.most_common(1)[0] if counts else ("None", 0)
            total = len(layer_results[l_idx])
            results_data[f"a{label:02d}"][f"l{l_idx:02d}"] = {
                "top": top_pair,
                "pct": round((count/total)*100, 1) if total > 0 else 0
            }

    with open('analysis_results.json', 'w') as f:
        json.dump(results_data, f, indent=4)
        
    print("DONE! Saved to analysis_results.json")

analyze_layers()
