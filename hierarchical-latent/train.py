import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
import logging
import time
from datetime import timedelta

from configs.config import ModelConfig
from models.hierarchical_vae import HierarchicalLatentDensity
from models.spatial_gnn import SpatialGNNDensity
from utils.data_loader import HierarchicalFeatureDataset, collate_features  # Add collate_features import

def validate_features(features, target_dims):
    """Validate feature shapes for debugging"""
    for i, (f, dim) in enumerate(zip(features, target_dims)):
        if not isinstance(f, torch.Tensor):
            raise TypeError(f"Feature {i} is not a tensor: {type(f)}")
        B, C, H, W = f.shape
        logging.info(f"Feature {i} shape: {f.shape}, target dim: {dim}")
        if C != dim:
            raise ValueError(f"Feature {i} has wrong channel dim: {C}, expected {dim}")

def process_features(features, projections, hidden_dims):
    """Process features maintaining spatial information"""
    processed_features = []
    spatial_info = []
    
    for feat, target_dim in zip(features, hidden_dims):
        # Keep track of original dimensions
        orig_shape = feat.shape
        B, C, H, W = orig_shape
        spatial_info.append((B, orig_shape))
        
        # Reshape to [B*H*W, C]
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C)
        processed_features.append(feat_flat)
        
        if len(processed_features) == 1:
            logging.info(f"Original shape: {orig_shape}, Flattened: {feat_flat.shape}")
    
    return processed_features, spatial_info

def train(config: ModelConfig):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Validate feature path
    feature_path = config.path
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature directory not found at {feature_path}")
        
    feature_files = list(feature_path.glob('*.pt'))
    if not feature_files:
        raise FileNotFoundError(f"No .pt files found in {feature_path}")
    logging.info(f"Found {len(feature_files)} feature files")
    
    # Initialize model with input type
    model = {
        "hierarchical_vae": HierarchicalLatentDensity,
        "spatial_gnn": SpatialGNNDensity
    }[config.model_type](config.level_dims, config.n_components, 
                        input_type=config.input_type)
    
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Create dataset and loader
    dataset = HierarchicalFeatureDataset(
        feature_path,
        target_dims=config.hidden_dims
    )
    
    # Get projection layers and move to GPU
    projections = dataset.get_projections('cuda')
    
    loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=collate_features,
        num_workers=0  # Disable multiprocessing for debugging
    )
    
    start_time = time.time()
    num_epochs = 100
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, features in enumerate(loader):
            optimizer.zero_grad()
            
            # Move to GPU and validate shapes
            features = [f.cuda() for f in features]
            
            if batch_idx == 0:
                logging.info("Before processing:")
                for i, f in enumerate(features):
                    logging.info(f"Level {i}: {f.shape}, channels: {f.size(1)}")
            
            # Process features with spatial information
            features, spatial_shapes = process_features(features, projections, config.hidden_dims)
            
            if batch_idx == 0:
                logging.info("After processing:")
                for i, f in enumerate(features):
                    logging.info(f"Level {i}: {f.shape}")
            
            # Process features and compute loss
            log_probs = model(features, spatial_shapes=spatial_shapes)
            loss = -sum(lp.mean() for lp in log_probs)
            
            # Log detailed loss info
            if batch_idx % 10 == 0:  # Every 10 batches
                batch_loss = loss.item()
                logging.info(f"Epoch {epoch}, Batch {batch_idx}: Loss = {batch_loss:.4f}")
                logging.info(f"Per-level log probs: {[lp.mean().item() for lp in log_probs]}")
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Log epoch summary
        epoch_loss = total_loss / num_batches
        logging.info(f"Epoch {epoch}: Average Loss = {epoch_loss:.4f}")
        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
        
        # Log epoch summary with timing
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        remaining_time = (epoch_time * (num_epochs - epoch - 1))
        
        logging.info(
            f"Epoch {epoch}/{num_epochs-1}: "
            f"Loss = {epoch_loss:.4f}, "
            f"Time = {timedelta(seconds=int(epoch_time))}, "
            f"Total = {timedelta(seconds=int(total_time))}, "
            f"Remaining ≈ {timedelta(seconds=int(remaining_time))}"
        )

if __name__ == "__main__":
    config = ModelConfig()
    train(config)
