"""
Emotion Dataset Management
--------------------------
Handles loading, splitting, and augmentation of the emotion classification dataset.
Supports:
- ImageFolder structure
- Standard augmentation (horizontal flip, color jitter)
- TrivialAugment (SOTA for limited data)
- Train/Validation splitting with leak-prevention
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import yaml


class EmotionDataModule:
    def __init__(self, config_source):
        if isinstance(config_source, str):
            with open(config_source, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_source
        
        self.data_dir = self.config['data']['data_dir']
        self.batch_size = self.config['data']['batch_size']
        self.num_workers = self.config['data']['num_workers']
        self.image_size = self.config['data']['image_size']
        self.val_split = self.config['data']['validation_split']

        # Augmentation settings
        aug_config = self.config.get('augmentation', {})
        use_trivial_augment = aug_config.get('trivial_augment', False)

        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
        ]
        
        if use_trivial_augment:
            # TrivialAugmentWide is SOTA for from-scratch training
            transform_list.append(transforms.TrivialAugmentWide())
        else:
            # Fallback to manual augmentation
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
            
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform = transforms.Compose(transform_list)

        self.val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_dataloaders(self):
        # Load full dataset
        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        
        # Calculate split sizes
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        
        # Split dataset
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Apply validation transform to validation set
        # Since RandomSplit returns Subsets that reference the original dataset, 
        # we cannot simply modify the transform of the subset's dataset without affecting training.
        # Instead, we use indices to create separate Subsets with distinct transforms.
        
        # Let's do the indices approach for correctness.
        generator = torch.Generator().manual_seed(self.config['training']['seed'])
        indices = torch.randperm(len(full_dataset), generator=generator).tolist()
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_subset = torch.utils.data.Subset(
            datasets.ImageFolder(root=self.data_dir, transform=self.transform),
            train_indices
        )
        
        val_subset = torch.utils.data.Subset(
            datasets.ImageFolder(root=self.data_dir, transform=self.val_transform),
            val_indices
        )

        loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': True,
        }
        
        if self.num_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = 4
            
        train_loader = DataLoader(
            train_subset, 
            shuffle=True, 
            **loader_kwargs
        )
        
        # Validation loader doesn't need shuffle or prefetch as much, but consistent is good
        val_loader = DataLoader(
            val_subset, 
            shuffle=False, 
            **loader_kwargs
        )
        
        return train_loader, val_loader, full_dataset.classes
