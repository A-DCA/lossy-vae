import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalLatentVAE(nn.Module):
    def __init__(self, input_dims, latent_dims=[64, 128, 256, 512]):
        super().__init__()
        self.latent_dims = latent_dims
        
        # Hierarchical encoder
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                # Mean and log variance for each level
                nn.Conv2d(out_dim, out_dim * 2, 1)
            ) for in_dim, out_dim in zip([input_dims] + latent_dims[:-1], latent_dims)
        ])
    
    def encode(self, x):
        """Encode input to hierarchical latent space"""
        latents = []
        current = x
        
        for encoder in self.encoders:
            # Get mean and log variance
            h = encoder(current)
            mu, logvar = h.chunk(2, dim=1)
            
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            latents.append((z, mu, logvar))
            current = z
            
        return latents
    
    def kl_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
