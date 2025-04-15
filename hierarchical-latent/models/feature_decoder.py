import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalDecoder(nn.Module):
    def __init__(self, level_dims):
        super().__init__()
        self.level_dims = level_dims
        
        # Create decoder blocks for each level
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // 2, 3, padding=1),
                nn.BatchNorm2d(dim // 2),
                nn.ReLU(),
                nn.Conv2d(dim // 2, dim // 2, 3, padding=1),
                nn.BatchNorm2d(dim // 2),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear')
            ) for dim in reversed(level_dims)
        ])
        
        # Final RGB decoder
        self.to_rgb = nn.Sequential(
            nn.Conv2d(level_dims[0] // 2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1),
            nn.Tanh()
        )
    
    def forward(self, hierarchical_features):
        """
        Progressively decode features from coarse to fine
        Args:
            hierarchical_features: List of feature tensors from different levels
        Returns:
            reconstructed_image: RGB image reconstruction
        """
        x = hierarchical_features[-1]  # Start with coarsest level
        
        # Progressive decoding
        for i, decoder in enumerate(self.decoders):
            x = decoder(x)
            if i < len(self.decoders) - 1:
                # Add features from finer level if available
                finer_features = hierarchical_features[-(i+2)]
                x = x + F.interpolate(finer_features, size=x.shape[-2:])
        
        # Generate final RGB image
        return self.to_rgb(x)

    def synthesize_from_features(self, features, temperature=1.0):
        """
        Synthesize image from sampled features
        Args:
            features: List of feature tensors or feature distributions
            temperature: Sampling temperature (higher = more variety)
        """
        decoded_features = []
        
        for feat in features:
            if isinstance(feat, torch.distributions.Distribution):
                # Sample from distribution
                sampled = feat.sample() * temperature
            else:
                # Use features directly
                sampled = feat
            decoded_features.append(sampled)
        
        return self.forward(decoded_features)
