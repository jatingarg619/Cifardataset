import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision import datasets

class CIFAR10Dataset:
    def __init__(self, root="./data", train=True):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        
        # Calculate dataset mean and std
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)
        
        self.train = train
        self.transforms = self.get_transforms()

    def get_transforms(self):
        if self.train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.CoarseDropout(
                    max_holes=1, max_height=16, max_width=16,
                    min_holes=1, min_height=16, min_width=16,
                    fill_value=self.mean, mask_fill_value=None,
                    p=0.5
                ),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        image = self.transforms(image=image)["image"]
        return image, label

    def __len__(self):
        return len(self.dataset) 