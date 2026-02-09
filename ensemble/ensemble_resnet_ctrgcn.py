import argparse
import pickle
import numpy as np
from tqdm import tqdm
import os

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def ensemble_fusion(label_path, resnet_path, ctrgcn_path, alpha=1.0):
    """
    label_path: Đường dẫn đến file nhãn thật (val_label.pkl)
    resnet_path: Đường dẫn đến file kết quả của ResNet (.pkl)
    ctrgcn_path: Đường dẫn đến file kết quả của CTR-GCN Joint (.pkl)
    alpha: Trọng số cho CTR-GCN (Mặc định là 1.0, tức là cộng ngang bằng: ResNet + 1.0 * CTRGCN)
    """
    try:
        label_data = load_pickle(label_path)
        label_data = np.array(label_data)
        print(f"Đã load Label từ: {label_path}")
    except Exception as e:
        print(f"Lỗi load file Label: {e}")
        return
    try:
        r_resnet = load_pickle(resnet_path)
        print(f"Đã load ResNet Score từ: {resnet_path}")
    except Exception as e:
        print(f"Lỗi load file ResNet: {e}")
        return
    try:
        r_ctrgcn = load_pickle(ctrgcn_path)
        print(f"Đã load CTR-GCN Score từ: {ctrgcn_path}")
    except Exception as e:
        print(f"Lỗi load file CTR-GCN: {e}")
        return
    right_num = 0
    total_num = 0
    sample_names = label_data[0]
    true_labels = label_data[1]

    print("Đang tiến hành Fusion (Ensemble)...")
    for i in tqdm(range(len(sample_names))):
        name = sample_names[i]
        l = int(true_labels[i])
        if name not in r_resnet or name not in r_ctrgcn:
            print(f"Cảnh báo: Không tìm thấy mẫu {name} trong file kết quả.")
            continue

        score_resnet = r_resnet[name]
        score_ctrgcn = r_ctrgcn[name]
        final_score = score_resnet + (alpha * score_ctrgcn)
    
        pred = np.argmax(final_score)
        if pred == l:
            right_num += 1
        
        total_num += 1
    acc = right_num / total_num
    print('\n' + '='*40)
    print(f'KẾT QUẢ ENSEMBLE (ResNet + Joint CTR-GCN)')
    print(f'Số mẫu đúng: {right_num}/{total_num}')
    print(f'Top-1 Accuracy: {acc:.4f} ({acc*100:.2f}%)')
    print('='*40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble ResNet + CTR-GCN')
    parser.add_argument('--label', default='./data/nucla/val_label.pkl', help='Path to val_label.pkl')
    parser.add_argument('--resnet', required=True, help='Path to ResNet score .pkl file')
    parser.add_argument('--ctrgcn', required=True, help='Path to CTR-GCN Joint score .pkl file')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for CTR-GCN score')
    args = parser.parse_args()
    ensemble_fusion(args.label, args.resnet, args.ctrgcn, args.alpha)