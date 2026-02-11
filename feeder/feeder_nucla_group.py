"""
Feeder cho ST-GCN 4-group classification.
Kế thừa data_dict từ feeder_nucla_gcn, chỉ đổi label mapping.

Map 10 UCLA action labels → 4 groups:
  Group 0: Walk around (4), Sit down (5), Stand up (6) — No object interaction
  Group 1: Pick up 1 hand (1), Pick up 2 hands (2), Drop trash (3), Throw (9) — Small object
  Group 2: Donning (7), Doffing (8) — Clothing
  Group 3: Carry (10) — Large object
"""
import numpy as np
import json
import random
import math
import os
from collections import Counter
from torch.utils.data import Dataset

# Map từ original label (1-indexed) → group label (0-indexed)
LABEL_TO_GROUP = {
    1: 1,   # pick up with one hand → Small Object
    2: 1,   # pick up with two hands → Small Object 
    3: 1,   # drop trash → Small Object
    4: 0,   # walk around → Body Motion
    5: 0,   # sit down → Body Motion
    6: 0,   # stand up → Body Motion
    7: 2,   # donning → Clothing
    8: 2,   # doffing → Clothing
    9: 1,   # throw → Small Object
    10: 3,  # carry → Large Object
}

GROUP_NAMES = {
    0: 'Body Motion (walk/sit/stand)',
    1: 'Small Object (pick/drop/throw)',
    2: 'Clothing (don/doff)',
    3: 'Large Object (carry)',
}

NUM_GROUPS = 4


class Feeder(Dataset):
    """Kế thừa data và preprocessing từ feeder_nucla_gcn, chỉ đổi label."""
    
    def __init__(self, data_path, label_path, repeat=1, random_choose=False, 
                 random_shift=False, random_move=False, window_size=-1, 
                 normalization=False, debug=False, use_mmap=True, evaluation=False):
        # Import data_dict từ feeder gốc
        from feeder.feeder_nucla_gcn import Feeder as OrigFeeder
        
        self.data_path = data_path
        self.label_path = label_path
        self.evaluation = evaluation
        
        # Tạo instance tạm để lấy data_dict
        orig = OrigFeeder.__new__(OrigFeeder)
        orig.data_path = data_path
        orig.label_path = label_path
        
        if 'val' in label_path:
            self.train_val = 'val'
        else:
            self.train_val = 'train'
        
        # Lấy data_dict trực tiếp bằng cách tạo instance
        temp = OrigFeeder(data_path=data_path, label_path=label_path, 
                         repeat=1, random_choose=False, random_shift=False,
                         random_move=False, window_size=window_size,
                         normalization=False, debug=debug)
        self.data_dict = temp.data_dict
        self.data = temp.data
        
        self.time_steps = 52
        
        # Map labels → group labels
        self.label = []
        self.original_label = []
        for info in self.data_dict:
            orig_lbl = int(info['label'])
            self.original_label.append(orig_lbl - 1)
            self.label.append(LABEL_TO_GROUP[orig_lbl])
        
        self.debug = debug
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.repeat = repeat
        
        # Print group distribution
        dist = Counter(self.label)
        print(f"  [{self.train_val}] Group distribution: {dict(sorted(dist.items()))}")
        for g, name in GROUP_NAMES.items():
            print(f"    Group {g}: {name} = {dist.get(g, 0)} samples")

    def __len__(self):
        return len(self.data_dict) * self.repeat

    def rand_view_transform(self, X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
        X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
        return np.reshape(X0, X.shape)

    def __getitem__(self, index):
        index = index % len(self.data_dict)
        label = self.label[index]
        value = self.data[index]

        if self.train_val == 'train':
            agx = random.randint(-60, 60)
            agy = random.randint(-60, 60)
            s = random.uniform(0.5, 1.5)
        else:
            agx, agy, s = 0, 0, 1.0

        center = value[0, 1, :]
        value = value - center
        scalerValue = self.rand_view_transform(value, agx, agy, s)
        scalerValue = np.reshape(scalerValue, (-1, 3))
        v_min, v_max = np.min(scalerValue, axis=0), np.max(scalerValue, axis=0)
        scalerValue = (scalerValue - v_min) / (v_max - v_min + 1e-6)
        scalerValue = scalerValue * 2 - 1
        scalerValue = np.reshape(scalerValue, (-1, 20, 3))

        data = np.zeros((self.time_steps, 20, 3))
        length = scalerValue.shape[0]

        if self.train_val == 'train':
            random_idx = random.sample(list(np.arange(length)) * 100, self.time_steps)
            random_idx.sort()
            data[:, :, :] = scalerValue[random_idx, :, :]
        else:
            idx = np.linspace(0, length - 1, self.time_steps).astype(int)
            data[:, :, :] = scalerValue[idx, :, :]

        data = np.transpose(data, (2, 0, 1))
        data = np.reshape(data, (3, self.time_steps, 20, 1))

        return data.astype(np.float32), label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
