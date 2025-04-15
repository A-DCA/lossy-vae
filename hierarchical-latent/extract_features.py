import torch
from pathlib import Path
from configs.config import ModelConfig
from utils.coco_feature_extractor import prepare_coco_features
from tqdm import tqdm

def main():
    config = ModelConfig()
    
    # Check if features already exist
    if not list(config.feature_save_dir.glob("*.pt")):
        print("Extracting features from COCO dataset...")
        prepare_coco_features(config)
    else:
        print("Features already extracted.")
        
    # Verify extraction
    feature_files = list(config.feature_save_dir.glob("*.pt"))
    print(f"Found {len(feature_files)} feature files")

if __name__ == "__main__":
    main()
