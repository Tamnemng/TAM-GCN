import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from feeder.feeder_nucla_gcn import Feeder as GCNFeeder
import random

class Feeder(GCNFeeder):
    def __init__(self, data_path, label_path, rgb_path, 
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False,
                 temporal_rgb_frames=5, random_flip=False, use_mmap=True, repeat=1, evaluation=False):
        
        # Initialize the parent GCN feeder, which sets up the skeletons, data dicts, and time_steps (52)
        super().__init__(data_path=data_path, label_path=label_path,
                         repeat=repeat, random_choose=random_choose, 
                         random_shift=random_shift, random_move=random_move,
                         window_size=window_size, normalization=normalization,
                         debug=debug, use_mmap=use_mmap)
        
        self.rgb_path = rgb_path
        self.temporal_rgb_frames = temporal_rgb_frames
        self.random_flip = random_flip
        
        # Identical transforms to the standalone ResNet implementation
        self.resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        # 1. Get skeleton data using the parent class
        # (This automatically applies view transforms/augmentations dynamically if train_val == 'train')
        data, _, label, idx = super().__getitem__(index)
        
        # 2. Load the corresponding unweighted base STROI image
        info = self.data_dict[index % len(self.data_dict)]
        filename = info['file_name']
        
        # The base STROI image without any weights!
        img_path = self.rgb_path + filename + '.png'
        
        try:
            rgb = Image.open(img_path).convert('RGB')
        except:
            print(f"Error loading base STROI image: {img_path}")
            rgb = Image.new('RGB', (224, 224))
            
        if self.train_val == 'train' and self.random_flip and random.random() < 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            
        rgb = self.resnet_transform(rgb)
        
        return data, rgb, label
