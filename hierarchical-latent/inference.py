import torch
from pathlib import Path
from configs.config import ModelConfig
from models.hierarchical_vae import HierarchicalLatentDensity
from models.spatial_gnn import SpatialGNNDensity
from utils.feature_extractor import HierarchicalFeatureExtractor
from utils.visualizer import FeatureVisualizer

def load_trained_model(checkpoint_path: Path, config: ModelConfig):
    """Load trained model from checkpoint"""
    # Initialize model
    model = {
        "hierarchical_vae": HierarchicalLatentDensity,
        "spatial_gnn": SpatialGNNDensity
    }[config.model_type](config.level_dims, config.n_components)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.cuda()

def estimate_feature_likelihood(model, image_path: Path, save_dir: Path):
    """Estimate likelihood of image features"""
    # Extract features
    extractor = HierarchicalFeatureExtractor(device='cuda')
    features = extractor.extract_features(image_path)
    
    # Get density estimates
    with torch.no_grad():
        log_probs = model([f.cuda() for f in features])
    
    # Visualize results
    visualizer = FeatureVisualizer(save_dir)
    visualizer.plot_hierarchical_features(features)
    visualizer.plot_density_estimates(log_probs)
    
    if isinstance(model, HierarchicalLatentDensity):
        # Get average log likelihood per level
        level_likelihoods = [lp.mean().item() for lp in log_probs]
        return level_likelihoods
    else:
        # For GNN model, return density scores
        return [lp.mean().item() for lp in log_probs]

def main():
    config = ModelConfig()
    checkpoint_path = Path("path/to/checkpoint.pt")  # Update this
    save_dir = Path("results")
    
    # Load model
    model = load_trained_model(checkpoint_path, config)
    
    # Process test images
    test_dir = Path("path/to/test/images")  # Update this
    results = {}
    
    for img_path in test_dir.glob("*.jpg"):
        likelihoods = estimate_feature_likelihood(model, img_path, save_dir)
        results[img_path.name] = likelihoods
        print(f"{img_path.name}: Level likelihoods = {likelihoods}")

if __name__ == "__main__":
    main()
