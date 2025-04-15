import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class SpatialGNNDensity(nn.Module):
    def __init__(self, level_dims, hidden_dim=256):
        super().__init__()
        self.level_dims = level_dims
        
        # Encoders for each level
        self.level_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for dim in level_dims
        ])
        
        # GNN layers for spatial message passing
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(len(level_dims))
        ])
        
        # Final density estimator
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def build_graph(self, hierarchical_features):
        """Convert hierarchical features to graph structure"""
        nodes = []
        edge_index = []
        level_indices = []
        
        offset = 0
        for level_idx, features in enumerate(hierarchical_features):
            B, C, H, W = features.shape
            level_features = features.permute(0,2,3,1).reshape(-1, C)
            
            # Add nodes
            nodes.append(level_features)
            level_indices.extend([level_idx] * (H*W))
            
            # Add edges between adjacent pixels
            for i in range(H*W):
                for j in range(i+1, H*W):
                    if self._are_adjacent(i, j, H, W):
                        edge_index.append([i+offset, j+offset])
                        edge_index.append([j+offset, i+offset])
            
            offset += H*W
        
        return (
            torch.cat(nodes, dim=0),
            torch.tensor(edge_index).t().contiguous(),
            torch.tensor(level_indices)
        )
    
    def _are_adjacent(self, i, j, H, W):
        """Check if two indices represent adjacent pixels"""
        i_row, i_col = i // W, i % W
        j_row, j_col = j // W, j % W
        return abs(i_row - j_row) + abs(i_col - j_col) == 1
    
    def forward(self, hierarchical_features):
        # Build graph from hierarchical features
        x, edge_index, level_indices = self.build_graph(hierarchical_features)
        
        # Initial node embeddings
        h = torch.zeros_like(x)
        for level in range(len(self.level_dims)):
            mask = level_indices == level
            h[mask] = self.level_encoders[level](x[mask])
        
        # Message passing
        for gnn in self.gnn_layers:
            h = gnn(h, edge_index)
        
        # Estimate density
        log_probs = self.density_head(h)
        
        # Reshape back to spatial form
        outputs = []
        offset = 0
        for features in hierarchical_features:
            B, C, H, W = features.shape
            level_size = H * W
            level_probs = log_probs[offset:offset+level_size]
            outputs.append(level_probs.reshape(B, H, W, 1))
            offset += level_size
            
        return outputs
