import io
import os
import glob
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# NTU-60 Cross-Subject split
TRAIN_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
# NTU-60 Cross-View split
TRAIN_CAMERAS = [2, 3]


_ZIP_CACHE = {}


def _get_zip(zip_path):
    key = (os.getpid(), zip_path)
    if key not in _ZIP_CACHE:
        _ZIP_CACHE[key] = zipfile.ZipFile(zip_path, 'r')
    return _ZIP_CACHE[key]


class Feeder(Dataset):
    def __init__(self, label_path, rgb_path,
                 split='CS',
                 random_flip=False,
                 debug=False,
                 **kwargs):
        """
        Args:
            label_path: 'train' or 'val'
            rgb_path: path to folder containing STROI .png files
            split: 'CS' (cross-subject) or 'CV' (cross-view)
        """
        self.label_path = label_path
        self.rgb_path = rgb_path
        self.split = split
        self.random_flip = random_flip
        self.debug = debug

        self.is_train = 'train' in label_path
        self._is_zip = rgb_path.endswith('.zip')

        if self._is_zip:
            with zipfile.ZipFile(rgb_path, 'r') as zf:
                self._zip_namelist = zf.namelist()
        else:
            self._zip_namelist = None

        self.load_data()

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def load_data(self):
        # Scan all .png files from a directory or zip archive.
        if self._is_zip:
            all_files = sorted([
                f for f in self._zip_namelist
                if f.lower().endswith('.png') and not f.startswith('__MACOSX')
                   and os.path.basename(f)
            ])
        else:
            all_files = sorted(glob.glob(os.path.join(self.rgb_path, '*.png')))
        if len(all_files) == 0:
            raise ValueError(f"No .png files found in {self.rgb_path}")

        self.data_dict = []
        for fpath in sorted(all_files):
            fname = os.path.splitext(os.path.basename(fpath))[0]
            # Parse: S013C001P007R001A057
            setup = int(fname[fname.find('S')+1 : fname.find('S')+4])
            camera = int(fname[fname.find('C')+1 : fname.find('C')+4])
            performer = int(fname[fname.find('P')+1 : fname.find('P')+4])
            action = int(fname[fname.find('A')+1 : fname.find('A')+4])

            # Determine train/test based on split
            if self.split == 'CS':
                is_train_sample = performer in TRAIN_SUBJECTS
            else:  # CV
                is_train_sample = camera in TRAIN_CAMERAS

            if self.is_train == is_train_sample:
                self.data_dict.append({
                    'file_name': fname,
                    'source_path': fpath,
                    'label': action - 1,  # 0-indexed
                })

        if self.debug:
            self.data_dict = self.data_dict[:100]

        print(f"NTU-60 {self.split} | {'train' if self.is_train else 'val'}: {len(self.data_dict)} samples")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        info = self.data_dict[index]
        filename = info['file_name']
        label = info['label']
        img_path = info['source_path']

        try:
            if self._is_zip:
                with _get_zip(self.rgb_path).open(img_path) as f:
                    rgb = Image.open(io.BytesIO(f.read())).convert('RGB')
            else:
                rgb = Image.open(img_path).convert('RGB')
        except:
            print(f"Error loading image: {img_path}")
            rgb = Image.new('RGB', (224, 224))

        rgb = self.transform(rgb)
        return rgb, label, filename
