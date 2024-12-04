from torch.utils.data import Dataset
from PIL import Image
import json
import os
import torch

class UIDataset(Dataset):
    def __init__(self, image_dir, annotations_dir, transform=None, max_features=500):
        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.max_features = max_features  # Maximum number of features to store
        
        # Exclude hidden files and only include images
        self.images = [f for f in sorted(os.listdir(image_dir))
                      if not f.startswith('.') and
                      f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Convert annotations to tensors
        self.annotations = []
        for img in self.images:
            ann_path = os.path.join(annotations_dir, os.path.splitext(img)[0] + '.json')
            with open(ann_path, 'r') as f:
                ann = json.load(f)
                
            # Extract numerical features recursively
            features = self.extract_features(ann)
            
            # Pad or truncate features to max_features
            if len(features) > self.max_features:
                features = features[:self.max_features]
            else:
                features.extend([0] * (self.max_features - len(features)))
                
            self.annotations.append(torch.tensor(features, dtype=torch.float))
    
    def extract_features(self, obj):
        """Recursively extract numerical features from JSON object"""
        features = []
        numerical_keys = ['x', 'y', 'width', 'height', 'rotation', 'opacity', 'cornerRadius']
        
        if isinstance(obj, dict):
            for key in numerical_keys:
                if key in obj:
                    features.append(float(obj[key]))
            
            # Recursively process children
            if 'children' in obj:
                for child in obj['children']:
                    features.extend(self.extract_features(child))
                    
        return features
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.annotations[idx]