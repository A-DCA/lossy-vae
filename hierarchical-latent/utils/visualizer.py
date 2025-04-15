import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE

class FeatureVisualizer:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_hierarchical_features(self, features, save_name="features.png"):
        """Plot features from different hierarchical levels"""
        n_levels = len(features)
        fig, axes = plt.subplots(1, n_levels, figsize=(5*n_levels, 5))
        
        for i, feat in enumerate(features):
            # Reduce dimensionality for visualization
            feat_flat = feat.reshape(-1, feat.size(1)).cpu().numpy()
            feat_2d = TSNE(n_components=2).fit_transform(feat_flat)
            
            axes[i].scatter(feat_2d[:, 0], feat_2d[:, 1], alpha=0.5)
            axes[i].set_title(f"Level {i+1}")
            
        plt.savefig(self.save_dir / save_name)
        plt.close()
        
    def plot_density_estimates(self, log_probs, save_name="density.png"):
        """Plot density estimates for each level"""
        n_levels = len(log_probs)
        fig, axes = plt.subplots(1, n_levels, figsize=(5*n_levels, 5))
        
        for i, probs in enumerate(log_probs):
            prob_map = torch.exp(probs).mean(-1)[0].cpu()
            sns.heatmap(prob_map, ax=axes[i], cmap='viridis')
            axes[i].set_title(f"Level {i+1} Density")
            
        plt.savefig(self.save_dir / save_name)
        plt.close()
        
    def plot_attention_maps(self, attention_weights, save_name="attention.png"):
        """Plot cross-level attention weights"""
        n_levels = len(attention_weights)
        fig, axes = plt.subplots(1, n_levels, figsize=(5*n_levels, 5))
        
        for i, weights in enumerate(attention_weights):
            sns.heatmap(weights[0].cpu(), ax=axes[i], cmap='viridis')
            axes[i].set_title(f"Level {i+1} Attention")
            
        plt.savefig(self.save_dir / save_name)
        plt.close()
