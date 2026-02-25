# sys
import os
import sys
import numpy as np
import random
import pickle
import json
import math
import re

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

# operation
from . import tools

class Feeder(Dataset):
    def __init__(self,
                 data_path,
                 label_path,
                 rgb_path,
                 random_choose=False,
                 random_move=False,
                 centralization=False,
                 window_size=-1,
                 part='train',
                 debug=False,
                 mmap=True,
                 random_interval=False,
                 random_roi_move=False,
                 temporal_rgb_frames=5,
                 evaluation=False,
                 random_flip=False):
                 
        self.debug = debug
        self.data_path = data_path
        self.rgb_path = rgb_path
        self.label_path = label_path # expects 'train' or 'val' inside config
        self.part = part
        if "val" in label_path:
            self.part = "val"
        self.temporal_rgb_frames = temporal_rgb_frames
        self.evaluation = (self.part != 'train')
        self.time_steps = 52

        # Reuse data dict and json loading from the GCN feeder
        import feeder.feeder_nucla_gcn as feeder_gcn
        temp_feeder = feeder_gcn.Feeder(data_path, label_path)
        self.data_dict = temp_feeder.data_dict
        self.label = temp_feeder.label
        self.data = temp_feeder.data

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.data_dict = self.data_dict[0:100]
            
        self.rgb_transform = transforms.Compose([
            transforms.Resize(size=(225, 45 * self.temporal_rgb_frames)), # Resize stitched image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.label)

    def rand_view_transform(self, X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
        X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __getitem__(self, index):
        index = index % len(self.data_dict)
        sample_info = self.data_dict[index]
        file_name = sample_info['file_name']
        label = self.label[index]
        value = self.data[index] # N, 3

        # SKELETON PREPROCESSING exactly like feeder_nucla_gcn.py
        if self.part == 'train':
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

        if self.part == 'train':
            random_idx = random.sample(list(np.arange(length)) * 100, self.time_steps)
            random_idx.sort()
            data[:, :, :] = scalerValue[random_idx, :, :]
        else:
            idx = np.linspace(0, length - 1, self.time_steps).astype(int)
            data[:, :, :] = scalerValue[idx, :, :]
            
        data = np.transpose(data, (2, 0, 1))
        data_numpy = np.reshape(data, (3, self.time_steps, 20, 1))

        # PRE-STITCHED RGB PROCESSING
        img_path = os.path.join(self.rgb_path, f"{file_name}.png")
        
        try:
            rgb_img = Image.open(img_path).convert('RGB')
            # optionally handle flip augmentation for training
            if self.part == 'train' and random.random() < 0.5:
                rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)
            rgb = self.rgb_transform(rgb_img)
        except Exception as e:
            # print(f"Error loading stitched image {img_path}: {e}")
            rgb = torch.zeros(3, 225, 45 * self.temporal_rgb_frames)
            
        return data_numpy.astype(np.float32), rgb, label
