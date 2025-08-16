"""
Simple Dataset Class for Waste Classification
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random


from utils.class_mapping import class_name_to_idx, idx_to_class_name


class WasteDataset(Dataset):
    """Simple dataset class for waste classification"""
    
    def __init__(self, data_dir, transform=None, mode='dummy', split='train'):
        """
        Initialize the dataset
        
        Args:
            data_dir (str): Path to data directory
            transform: Image transformations
            mode (str): 'dummy' for generated data, 'real' for actual images
            split (str): 'train' or 'val'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.split = split
        
        # Use class mapping from utils
        self.class_to_idx = class_name_to_idx
        self.idx_to_class = idx_to_class_name
        self.classes = list(class_name_to_idx.keys())
        self.num_classes = len(self.classes)
        
        # Load samples
        if mode == 'dummy':
            self.samples = self._generate_dummy_samples()
        else:
            self.samples = self._load_real_samples()
    
    def _generate_dummy_samples(self):
        """Generate dummy samples for testing"""
        num_samples = 800 if self.split == 'train' else 200
        samples = []
        
        for i in range(num_samples):
            class_idx = random.randint(0, self.num_classes - 1)
            class_name = self.classes[class_idx]
            image_path = f"dummy_{self.split}_{i}_{class_name}.jpg"
            samples.append((image_path, class_idx))
        
        return samples
    
    def _load_real_samples(self):
        """Load real image samples from directory structure"""
        samples = []
        
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist. Using dummy mode.")
            return self._generate_dummy_samples()
        
        # Load from class directories
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                class_idx = self.class_to_idx[class_name]
                
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_file)
                        samples.append((img_path, class_idx))
        
        if not samples:
            print("No real images found. Using dummy mode.")
            return self._generate_dummy_samples()
        
        return samples
    
    def _generate_dummy_image(self):
        """Generate a simple dummy RGB image"""
        # Create a simple 224x224 RGB image with random noise
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(img)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        img_path, class_idx = self.samples[idx]
        
        if self.mode == 'dummy':
            image = self._generate_dummy_image()
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image = self._generate_dummy_image()
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, class_idx


def get_transforms(image_size=224, is_training=True):
    """Get simple data transformations"""
    if is_training:
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    return transforms.Compose(transform_list)


def create_data_loaders(data_dir, batch_size=32, num_workers=4, mode='dummy'):
    """Create training and validation data loaders"""
    
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Create datasets
    train_dataset = WasteDataset(
        data_dir=data_dir,
        transform=train_transform,
        mode=mode,
        split='train'
    )
    
    val_dataset = WasteDataset(
        data_dir=data_dir,
        transform=val_transform,
        mode=mode,
        split='val'
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Simple test
    print("Testing WasteDataset...")
    
    train_loader, val_loader = create_data_loaders('./data', batch_size=4, num_workers=0, mode='dummy')
    
    print(f"Train dataset: {len(train_loader.dataset)} samples")
    print(f"Val dataset: {len(val_loader.dataset)} samples")
    print(f"Classes: {train_loader.dataset.classes}")
    
    # Test one batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels: {labels}")
        break
    
    print("Test completed!")