import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

class UIDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        
        # Get list of all images
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load annotation
        ann_name = os.path.splitext(img_name)[0] + '.json'
        ann_path = os.path.join(self.annotation_dir, ann_name)
        
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
        
        # Convert annotation to tensor
        # Assuming annotation contains x, y, width, height
        target = torch.tensor([
            annotation['x'],
            annotation['y'],
            annotation['width'],
            annotation['height']
        ], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, target
