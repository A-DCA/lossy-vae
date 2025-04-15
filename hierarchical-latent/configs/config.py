from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    model_type: str = "hierarchical_vae"  # or "spatial_gnn"
    input_type: str = "features"  # Type of input: "features" or "images"

    level_dims: List[int] = None
    hidden_dim: int = 256
    hidden_dims: List[int] = None  # Hidden dimensions for each level
    n_components: int = 8
    learning_rate: float = 1e-4
    batch_size: int = 32
    
    # COCO dataset paths
    coco_root: Path = Path("/mnt/e/lossy-vae/datasets/coco")
    coco_train: str = "train2017"
    ann_file: Path = Path("/mnt/e/lossy-vae/datasets/coco/annotations/instances_train2017.json")
    dataset_dir: Path = Path("/mnt/e/lossy-vae/datasets/coco/features/train2017")
    
    @property
    def path(self) -> Path:
        """Path to the feature directory used by HierarchicalFeatureDataset"""
        if not self.dataset_dir.exists():
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
        return self.dataset_dir.resolve()  # Return absolute path
    
    @path.setter
    def path(self, value: Path):
        self.dataset_dir = Path(value)
    
    def __post_init__(self):
        if self.level_dims is None:
            self.level_dims = [256, 512, 1024, 2048]  # ResNet50 feature dimensions
        if self.hidden_dims is None:
            # Use same hidden dimension for all levels
            self.hidden_dims = [self.hidden_dim] * len(self.level_dims)
        
        # Validate paths
        if not self.ann_file.exists():
            raise FileNotFoundError(
                f"COCO annotations not found at {self.ann_file}. "
                "Please download COCO annotations and place them in the correct directory.\n"
                "You can download them from: http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            )
        
        image_dir = self.coco_root / self.coco_train
        if not image_dir.exists():
            raise FileNotFoundError(
                f"COCO images not found at {image_dir}. "
                "Please download COCO train2017 images and place them in the correct directory.\n"
                "You can download them from: http://images.cocodataset.org/zips/train2017.zip"
            )
        
        self.dataset_dir = Path(self.dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
