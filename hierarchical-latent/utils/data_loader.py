import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

def collate_features(batch):
    """Custom collate ensuring 4D tensors"""
    collated = []
    for level in zip(*batch):
        # Ensure 4D features [B, C, H, W]
        level_tensors = []
        for feat in level:
            if len(feat.shape) == 3:  # [C, H, W]
                feat = feat.unsqueeze(0)
            level_tensors.append(feat)
            
        stacked = torch.cat(level_tensors, dim=0)
        logging.info(f"Collated feature shape: {stacked.shape}")
        collated.append(stacked)
    return collated

class HierarchicalFeatureDataset(Dataset):
    def __init__(self, feature_dir, target_dims=None):
        super().__init__()
        self.feature_dir = Path(feature_dir)
        self.feature_files = sorted(list(self.feature_dir.glob('*.pt')))
        self.target_dims = target_dims  # List of target dimensions for each level
        self.projections = {}  # Store projections as module dict
        
        # Validate first file
        if self.feature_files:
            sample = torch.load(self.feature_files[0])
            if not isinstance(sample, list):
                raise ValueError(f"Expected list of features, got {type(sample)}")
            self.num_levels = len(sample)
            logging.info(f"Found {self.num_levels} feature levels")
            
            if self.target_dims and len(self.target_dims) != self.num_levels:
                raise ValueError("Number of target dimensions must match number of feature levels")
    
    def __len__(self):
        return len(self.feature_files)
        
    def __getitem__(self, idx):
        features = torch.load(self.feature_files[idx])
        # Return raw features without dimension adjustment
        return [f if len(f.shape) == 3 else f.squeeze(0) for f in features]
    
    def get_projections(self, device):
        """Create projection layers for dimension adjustment"""
        if not self.target_dims:
            return None
            
        sample = torch.load(self.feature_files[0])
        for level, (feat, target_dim) in enumerate(zip(sample, self.target_dims)):
            if feat.size(1) != target_dim:
                key = f'proj_{feat.size(1)}_{target_dim}'
                if key not in self.projections:
                    self.projections[key] = torch.nn.Conv2d(feat.size(1), target_dim, 1).to(device)
        return self.projections
