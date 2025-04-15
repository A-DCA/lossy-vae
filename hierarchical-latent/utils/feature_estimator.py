import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from utils.feature_extractor import HierarchicalFeatureExtractor
from utils.visualizer import FeatureVisualizer
from models.feature_decoder import HierarchicalDecoder

class FeatureEstimator:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.extractor = HierarchicalFeatureExtractor(device=device)
        self.visualizer = FeatureVisualizer(Path("results"))
        self.decoder = HierarchicalDecoder(model.level_dims).to(device)

    def estimate_pixel_features(self, image_path, level_idx=None):
        """
        Estimate feature distributions at each pixel location
        Args:
            image_path: Path to image
            level_idx: Which hierarchical level to analyze (None for all levels)
        Returns:
            Dictionary with:
            - log_probs: Log probabilities at each spatial location
            - most_likely_features: Most likely feature vectors
            - uncertainty: Uncertainty maps
        """
        # Extract features
        with torch.no_grad():
            features = self.extractor.extract_features(image_path)
            features = [f.to(self.device) for f in features]
            
            # Get density estimates
            log_probs = self.model(features)
            
            results = {}
            levels = [level_idx] if level_idx is not None else range(len(features))
            
            for idx in levels:
                # Get spatial dimensions
                B, H, W, C = log_probs[idx].shape
                
                # Get most likely mixture component at each location
                max_probs, max_indices = log_probs[idx].max(dim=-1)
                
                # Get corresponding feature vectors
                if hasattr(self.model, 'level_estimators'):  # For VAE model
                    components = self.model.level_estimators[idx][0].locs  # Get mixture components
                    most_likely = components[max_indices]
                else:  # For GNN model
                    most_likely = features[idx].permute(0, 2, 3, 1)
                
                # Calculate uncertainty (entropy of probabilities)
                probs = torch.exp(log_probs[idx])
                entropy = -(probs * log_probs[idx]).sum(dim=-1)
                
                results[f'level_{idx}'] = {
                    'log_probs': log_probs[idx].cpu(),
                    'most_likely_features': most_likely.cpu(),
                    'uncertainty': entropy.cpu(),
                    'spatial_shape': (H, W)
                }
            
            return results

    def interpolate_features(self, pos1, pos2, level_idx, num_steps=10):
        """Interpolate between two spatial locations in feature space"""
        results = []
        for alpha in np.linspace(0, 1, num_steps):
            interp_pos = tuple(alpha * p1 + (1-alpha) * p2 for p1, p2 in zip(pos1, pos2))
            results.append(self.estimate_pixel_features(interp_pos, level_idx))
        return results

    def synthesize_from_location(self, image_path, pixel_coords, level_idx=None):
        """Synthesize image using features from specific location"""
        results = self.estimate_pixel_features(image_path, level_idx)
        
        synthesized_images = []
        for level, data in results.items():
            features = data['most_likely_features']
            h, w = data['spatial_shape']
            y, x = pixel_coords
            
            # Get features at specified location
            loc_features = features[:, y, x]
            
            # Synthesize using decoder
            synth = self.decoder.synthesize_from_features([loc_features])
            synthesized_images.append(synth)
            
        return synthesized_images

    def interpolate_and_synthesize(self, pos1, pos2, level_idx, num_steps=10):
        """Interpolate between locations and synthesize images"""
        interp_features = self.interpolate_features(pos1, pos2, level_idx, num_steps)
        return [self.decoder.synthesize_from_features(feat) for feat in interp_features]

def main():
    # Example usage
    from models.hierarchical_vae import HierarchicalLatentDensity
    from configs.config import ModelConfig
    
    config = ModelConfig()
    model = HierarchicalLatentDensity(config.level_dims, config.n_components)
    model.load_state_dict(torch.load("checkpoint.pt")['model_state_dict'])
    
    estimator = FeatureEstimator(model)
    
    # Get feature estimates for an image
    image_path = Path("test_image.jpg")
    results = estimator.estimate_pixel_features(image_path)
    
    # Analyze specific level
    level_2_results = estimator.estimate_pixel_features(image_path, level_idx=2)
    
    # Visualize results
    for level, data in results.items():
        print(f"\nLevel {level}:")
        print(f"Feature shape: {data['most_likely_features'].shape}")
        print(f"Spatial dimensions: {data['spatial_shape']}")
        print(f"Average uncertainty: {data['uncertainty'].mean().item():.4f}")

if __name__ == "__main__":
    main()
