"""
Simple Dataset Class for Waste Classification
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image

from utils.class_mapping import class_name_to_idx, idx_to_class_name


class WasteDataset_v2(Dataset):
    """Simple dataset class for waste classification"""
    
    def __init__(self, data_dir, split_version='v1', split='train'):
        """
        Initialize the dataset
        
        Args:
            data_dir (str): Path to data directory
            split (str): 'train' or 'val'
        """
        self.data_dir = data_dir
        self.split_version = split_version
        self.transform = self._init_transform(split == 'train')
        self.split = split
        
        # Use class mapping from utils
        self.class_to_idx = class_name_to_idx
        self.idx_to_class = idx_to_class_name
        self.classes = list(class_name_to_idx.keys())
        self.num_classes = len(self.classes)
        
        # Load samples
        self.samples = self._load_real_samples()

    def _init_transform(self, is_training=True):
        """Initialize transformations"""
        transform = get_transforms(is_training)
        return transform


    
    def _load_real_samples(self):
        """Load real image samples from directory structure"""
        samples = []
        split_name = f"{self.split}_{self.split_version}.txt"
        split_path = os.path.join(self.data_dir, split_name)
        
        if not os.path.exists(split_path):
            print(f"Split file {split_path} does not exist.")
            return []

        with open(split_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                img_path, class_idx = line[:-2].strip(), int(line[-1].strip())
                samples.append((os.path.join(self.data_dir, img_path), class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        img_path, class_idx = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, class_idx


def get_transforms(is_training=True):
    """Get simple data transformations"""
    if is_training:
        transform_list = [
            transforms.Resize(360),
            transforms.RandomResizedCrop(320, scale=(0.6,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(10),
            # transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            v2.GaussianNoise(mean=0.0, sigma=1. / 255., clip=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        transform_list = [
            transforms.Resize(360),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            v2.GaussianNoise(mean=0.0, sigma=1. / 255., clip=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    return transforms.Compose(transform_list)

def create_data_loaders(dataset_config, worker_init_fn=None):
    """Create training and validation data loaders"""
    
    # Create datasets
    train_dataset = WasteDataset_v1(
        data_dir=dataset_config['data_dir'],
        split_version=dataset_config['split_version'],
        split='train'
    )
    
    val_dataset = WasteDataset_v1(
        data_dir=dataset_config['data_dir'],  
        split_version=dataset_config['split_version'],
        split='val'
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=True,
        num_workers=dataset_config['num_workers'],
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=False,
        num_workers=dataset_config['num_workers'],
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )
    
    return train_loader, val_loader