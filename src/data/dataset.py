import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ParkinsonDataset(Dataset):
    def __init__(self, root_dir, transform=None, enhance=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            enhance (bool): Apply the custom YUV + Histogram Equalization enhancement.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.enhance = enhance
        self.image_paths = []
        self.labels = []
        
        # Load images (Assuming structured folders: 'PD' and 'Control')
        for label, class_name in enumerate(['Control', 'PD']):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def _enhance_image(self, image):
        """
        Your custom enhancement pipeline: RGB -> YUV -> CLAHE on Y -> RGB
        """
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(img_yuv)
        
        # Apply Gaussian Blur to Y channel
        y_blurred = cv2.GaussianBlur(y, (3, 3), 0)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        y_eq = clahe.apply(y_blurred)
        
        # Merge and convert back
        img_yuv_eq = cv2.merge((y_eq, u, v))
        return cv2.cvtColor(img_yuv_eq, cv2.COLOR_YUV2RGB)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Ensure RGB
        
        if self.enhance:
            image = self._enhance_image(image)
            
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            image = transforms.ToTensor()(image)
            
        return image, self.labels[idx]