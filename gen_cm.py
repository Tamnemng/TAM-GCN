import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import feeder.feeder_nucla_fusion as feeder
import os

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# The result is saved as test_result.pkl
result_file = 'test_result.pkl'
if not os.path.exists(result_file):
    print("test_result.pkl not found!")
    exit(1)

results = load_pkl(result_file)

# load val data dict to get ground truth
f = feeder.Feeder(split='val')
data_dict = f.data_dict

true_labels = []
pred_labels = []

for item in data_dict:
    name = item['file_name']
    label = item['label'] - 1 # 0-indexed
    if name in results:
        true_labels.append(label)
        pred_labels.append(results[name])
    else:
        print(f"Missing {name} in predictions!")

if len(true_labels) > 0:
    cm = confusion_matrix(true_labels, pred_labels)
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"Evaluation Accuracy from PKL: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    
    # Save the confusion matrix to a file so it can be viewed nicely
    np.savetxt("confusion_matrix.txt", cm, fmt="%d")
else:
    print("No valid results matched!")
