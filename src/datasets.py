import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MultimodalDataset(Dataset):
    def __init__(self, root_dir, rgb_folder='rgb', lidar_folder='lidar', subset=1.0, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] 

        classes = ['cubes', 'spheres']
        class_to_idx = {'cubes': 0, 'spheres': 1}

        print(f"Scanning dataset at {root_dir}...")

        for cls_name in classes:
            cls_rgb_dir = os.path.join(root_dir, cls_name, rgb_folder)
            cls_lidar_dir = os.path.join(root_dir, cls_name, lidar_folder)
            
            if not os.path.exists(cls_rgb_dir):
                print(f"⚠️ Warning: Could not find {cls_rgb_dir}. Skipping.")
                continue
            
            rgb_files = sorted([f for f in os.listdir(cls_rgb_dir) if f.endswith('.png')])
            lidar_files = sorted([f for f in os.listdir(cls_lidar_dir) if f.endswith('.npy')])
            
            # Truncate mismatch
            if len(rgb_files) != len(lidar_files):
                min_len = min(len(rgb_files), len(lidar_files))
                rgb_files = rgb_files[:min_len]
                lidar_files = lidar_files[:min_len]

            label = class_to_idx[cls_name]
            for r, l in zip(rgb_files, lidar_files):
                self.samples.append({
                    'rgb': os.path.join(cls_rgb_dir, r),
                    'lidar': os.path.join(cls_lidar_dir, l),
                    'label': label
                })

        print(f"✅ Loaded {len(self.samples)} total samples.")

        # --- THE FIX: SHUFFLE BEFORE SUBSET ---
        # This ensures we get a mix of Cubes and Spheres, not just the first 10% (Cubes)
        random.seed(42) # Fixed seed for reproducibility
        random.shuffle(self.samples) 

        # Subset Logic
        if subset < 1.0:
            count = int(len(self.samples) * subset)
            self.samples = self.samples[:count]
            print(f"✂️  Subset active: Keeping random {count} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        rgb_img = Image.open(item['rgb']).convert("RGBA")
        rgb_tensor = transforms.ToTensor()(rgb_img)
        
        lidar_data = np.load(item['lidar']).astype(np.float32)
        lidar_tensor = torch.from_numpy(lidar_data)
        
        if lidar_tensor.ndim == 2:
            lidar_tensor = lidar_tensor.unsqueeze(0)
            
        return rgb_tensor, lidar_tensor, torch.tensor(item['label'])