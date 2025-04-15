import torch
import torch.nn as nn
import torch.distributions as dist
import logging

class MixtureModule(nn.Module):
    def __init__(self, input_dim, n_components):
        super().__init__()
        self.input_dim = input_dim
        self.n_components = n_components
        self.gmm_dim = 64
        
        # Two-stage dimension reduction
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, self.gmm_dim)
        )
        
        # GMM parameters
        self.mix_logits = nn.Parameter(torch.zeros(n_components))
        self.locs = nn.Parameter(torch.randn(n_components, self.gmm_dim))
        self.scale_tril = nn.Parameter(torch.eye(self.gmm_dim).unsqueeze(0).repeat(n_components,1,1))
        
        # Add debug mode
        self.debug = True
        self.debug_counter = 0
    
    def forward(self, x):
        # Validate input
        if self.debug and self.debug_counter % 100 == 0:
            self.debug_counter += 1
            logging.info(f"MixtureModule stats:\n"
                        f"Input shape: {x.shape}\n"
                        f"Input mean: {x.mean():.3f}, std: {x.std():.3f}\n"
                        f"Input range: [{x.min():.3f}, {x.max():.3f}]")
        
        # Input shape debug
        if self.training and torch.rand(1) < 0.01:
            logging.info(f"MixtureModule input: {x.shape}")
        
        # Project features
        x_proj = self.proj(x)  # [N, gmm_dim]
        
        # GMM estimation
        mixture_dist = dist.Categorical(logits=self.mix_logits)
        comp_dist = dist.MultivariateNormal(self.locs, scale_tril=self.scale_tril)
        gmm = dist.MixtureSameFamily(mixture_dist, comp_dist)
        
        # Get scalar log probabilities [N]
        return gmm.log_prob(x_proj)

class HierarchicalLatentDensity(nn.Module):
    def __init__(self, level_dims, n_components=8, input_type='features'):
        super().__init__()
        self.level_dims = level_dims
        self.input_type = input_type
        
        if input_type == 'latents':
            self.latent_vae = HierarchicalLatentVAE(3, level_dims)
        
        # Create level estimators
        self.level_estimators = nn.ModuleList([
            MixtureModule(dim, n_components)
            for dim in level_dims
        ])
        
        # Cross-level attention for spatial relationships
        self.level_attention = nn.ModuleList([
            nn.MultiheadAttention(dim, 4)
            for dim in level_dims[:-1]
        ])
        
        # Add logging setup
        self.logger = logging.getLogger(__name__)

    def forward(self, features, spatial_shapes=None):
        """
        Args:
            features: List of tensors [(N, C1)...] where N = B*H*W
            spatial_shapes: List of tuples [(B, shape)...] containing original shapes
        """
        if self.input_type == 'latents':
            hierarchical_features = [z for z, _, _ in self.latent_vae.encode(features)]
        else:
            hierarchical_features = features
        
        log_probs = []
        
        for level_idx, (feat, (batch_size, orig_shape)) in enumerate(zip(hierarchical_features, spatial_shapes)):
            # Get density estimation (returns [N] shaped tensor)
            level_log_probs = self.level_estimators[level_idx](feat)
            
            # Reshape to match original spatial dimensions [B, H, W]
            level_log_probs = level_log_probs.view(orig_shape[0], orig_shape[2], orig_shape[3])
            log_probs.append(level_log_probs)
        
        return log_probs
