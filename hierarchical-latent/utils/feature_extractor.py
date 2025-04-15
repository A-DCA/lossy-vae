import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms, models
import numpy as np
from typing import Optional

class HierarchicalFeatureExtractor:
    def __init__(self, pretrained_model_path: Optional[Path] = None, device='cuda'):
        self.device = device
        self.model = self.load_pretrained_model(pretrained_model_path)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def load_pretrained_model(self, model_path):
        # Load pretrained ResNet50
        model = models.resnet50(pretrained=True)
        
        # Load custom weights if path is provided and exists
        if model_path is not None:
            if model_path.exists():
                model.load_state_dict(torch.load(model_path))
            else:
                print(f"Warning: Model path {model_path} does not exist. Using default pretrained weights.")
            
        return model.to(self.device)
    
    def extract_features(self, image_path: Path):
        """Extract hierarchical features from different layers"""
        features = []
        hooks = []
        
        def hook_fn(module, input, output):
            features.append(output.detach())
            
        # Add hooks to desired layers
        hook_layers = [
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4
        ]
        
        for layer in hook_layers:
            hooks.append(layer.register_forward_hook(hook_fn))
            
        # Process image
        img = self.transform(image_path).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model(img)
            
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return features

    def extract_batch_features(self, images):
        """Extract features from a batch of images more efficiently"""
        features = []
        hooks = []
        
        def hook_fn(module, input, output, level_idx=None):
            # Store features with level index to maintain order
            features.append((level_idx, output.detach()))
        
        # Register hooks once for all layers
        hook_layers = [
            (0, self.model.layer1),
            (1, self.model.layer2),
            (2, self.model.layer3),
            (3, self.model.layer4)
        ]
        
        for idx, layer in hook_layers:
            hooks.append(
                layer.register_forward_hook(
                    lambda m, i, o, idx=idx: hook_fn(m, i, o, idx)
                )
            )
        
        try:
            # Process batch
            with torch.no_grad():
                self.model(images)
            
            # Sort features by level index
            features.sort(key=lambda x: x[0])
            return [f[1] for f in features]
            
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()

def extract_and_save_features(image_dir: Path, output_dir: Path, extractor: HierarchicalFeatureExtractor):
    """Extract and save features for a directory of images"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in image_dir.glob('*.jpg'):
        features = extractor.extract_features(img_path)
        output_path = output_dir / f"{img_path.stem}_features.pt"
        torch.save(features, output_path)
