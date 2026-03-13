"""
NTU RGB+D 60 Feeder for On-The-Fly ResNet + CTR-GCN fusion.

Loads:
  1. Skeleton data from .npz file (x_train/x_test, y_train/y_test)
  2. ST-ROI images from a folder of .png files (S013C001P007R001A057.png)

Matching strategy:
  - Scan all .png files in rgb_path, sort alphabetically
  - Split by CV (camera 2,3 = train) or CS (subject-based)
  - The resulting ordered list matches the npz sample ordering
    (since both were generated from the same sorted .skeleton files)
  - If npz was cleaned (some zero-samples removed), use greedy label matching
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# NTU-60 Cross-Subject split
TRAIN_SUBJECTS_CS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
# NTU-60 Cross-View split
TRAIN_CAMERAS_CV = [2, 3]


class Feeder(Dataset):
    def __init__(self, npz_path, rgb_path, label_path='train',
                 split='CV', time_steps=64, temporal_rgb_frames=5,
                 random_flip=False, random_choose=False, debug=False,
                 **kwargs):
        """
        Args:
            npz_path: path to NTU60_CV_CLEAN.npz or NTU60_CS_CLEAN.npz
            rgb_path: path to folder containing ST-ROI .png files
            label_path: 'train' or 'val'
            split: 'CV' (cross-view) or 'CS' (cross-subject)
            time_steps: temporal window for skeleton (default 64)
            temporal_rgb_frames: number of temporal frames in skeleton grid (default 5)
            random_flip: random horizontal flip for RGB augmentation
        """
        self.rgb_path = rgb_path
        self.split = split
        self.time_steps = time_steps
        self.temporal_rgb_frames = temporal_rgb_frames
        self.random_flip = random_flip
        self.is_train = 'train' in label_path
        self.debug = debug

        # Load skeleton data from npz
        self._load_npz(npz_path)

        # Build image path mapping
        self._build_image_mapping()

        # RGB transforms
        if self.is_train:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _load_npz(self, npz_path):
        """Load skeleton data and labels from npz."""
        data = np.load(npz_path)
        if self.is_train:
            self.skeleton_data = data['x_train']    # (N, T_max, 150)
            labels_onehot = data['y_train']         # (N, 60)
        else:
            self.skeleton_data = data['x_test']
            labels_onehot = data['y_test']

        self.labels = labels_onehot.argmax(axis=1)  # (N,) — class index 0-59
        print(f"NTU-60 {self.split} | {'train' if self.is_train else 'val'}: "
              f"{len(self.labels)} skeleton samples loaded from npz")

    def _build_image_mapping(self):
        """Match ST-ROI images to skeleton samples by split + label ordering."""
        all_pngs = sorted(glob.glob(os.path.join(self.rgb_path, '*.png')))
        if len(all_pngs) == 0:
            raise ValueError(f"No .png files found in {self.rgb_path}")

        # Split images using same logic as npz generation
        my_images = []
        my_labels = []
        for fpath in all_pngs:
            fname = os.path.splitext(os.path.basename(fpath))[0]
            camera = int(fname[fname.find('C') + 1: fname.find('C') + 4])
            performer = int(fname[fname.find('P') + 1: fname.find('P') + 4])
            action = int(fname[fname.find('A') + 1: fname.find('A') + 4]) - 1  # 0-indexed

            if self.split == 'CV':
                is_train_sample = camera in TRAIN_CAMERAS_CV
            else:  # CS
                is_train_sample = performer in TRAIN_SUBJECTS_CS

            if self.is_train == is_train_sample:
                my_images.append(fpath)
                my_labels.append(action)

        n_skel = len(self.labels)
        n_img = len(my_images)

        if n_skel == n_img:
            # Perfect match — verify labels agree
            mismatches = sum(1 for i in range(n_skel) if my_labels[i] != self.labels[i])
            if mismatches == 0:
                self.image_paths = my_images
                print(f"  Image matching: perfect 1:1 match ({n_img} images)")
                return
            else:
                print(f"  WARNING: {mismatches}/{n_skel} label mismatches with direct mapping")

        # npz was cleaned (fewer skeletons than images) — greedy label match
        print(f"  Sample count: {n_img} images vs {n_skel} skeletons. "
              f"Using greedy label matching...")
        matched_images = []
        skel_idx = 0
        for img_idx in range(n_img):
            if skel_idx >= n_skel:
                break
            if my_labels[img_idx] == self.labels[skel_idx]:
                matched_images.append(my_images[img_idx])
                skel_idx += 1

        if skel_idx == n_skel:
            self.image_paths = matched_images
            print(f"  Greedy match: successfully matched all {n_skel} skeleton samples")
        else:
            raise ValueError(
                f"Could not match all skeleton samples to images. "
                f"Matched {skel_idx}/{n_skel}. "
                f"Check that ST-ROI images and npz come from the same data source."
            )

        if self.debug:
            self.skeleton_data = self.skeleton_data[:100]
            self.labels = self.labels[:100]
            self.image_paths = self.image_paths[:100]

    def _temporal_sample(self, skeleton):
        """Sample skeleton to fixed time_steps.

        Args:
            skeleton: (T_max, 150) — zero-padded

        Returns:
            (time_steps, 150) — temporally normalized
        """
        # Find actual length (last non-zero frame)
        frame_sums = np.abs(skeleton).sum(axis=1)
        nonzero_frames = np.where(frame_sums > 0)[0]

        if len(nonzero_frames) == 0:
            return np.zeros((self.time_steps, 150), dtype=np.float32)

        actual_len = nonzero_frames[-1] + 1
        skeleton = skeleton[:actual_len]

        if self.is_train:
            # Random sampling with repetition
            indices = sorted(random.choices(range(actual_len), k=self.time_steps))
        else:
            # Uniform interpolation
            indices = np.linspace(0, actual_len - 1, self.time_steps).astype(int)

        return skeleton[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 1. Load and process skeleton
        skel_raw = self.skeleton_data[index]                 # (T_max, 150)
        skel_sampled = self._temporal_sample(skel_raw)       # (time_steps, 150)

        # Reshape: (T, 150) → (T, 2, 25, 3) → (3, T, 25, 2) = (C, T, V, M)
        T = self.time_steps
        skel = skel_sampled.reshape(T, 2, 25, 3)            # (T, M, V, C)
        skel = skel.transpose(3, 0, 2, 1)                   # (C, T, V, M) = (3, T, 25, 2)
        skel = skel.astype(np.float32)

        # 2. Load ST-ROI image
        img_path = self.image_paths[index]
        try:
            rgb = Image.open(img_path).convert('RGB')
        except Exception:
            print(f"Error loading image: {img_path}")
            rgb = Image.new('RGB', (224, 224))

        if self.is_train and self.random_flip and random.random() < 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)

        rgb = self.rgb_transform(rgb)

        label = int(self.labels[index])
        return skel, rgb, label
