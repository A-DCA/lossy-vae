import torch
from PIL import Image
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from .feature_extractor import HierarchicalFeatureExtractor

def custom_collate(batch):
    """Custom collate function to handle batching"""
    images = []
    targets = []
    for img, target in batch:
        if img is not None:
            images.append(img)
            targets.append(target)
    
    if not images:
        return None, None
    return torch.stack(images), targets

class CocoDatasetWrapper(CocoDetection):
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = str(Path(self.root) / img_info['file_name'])
        
        try:
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, img_id
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None, None

def prepare_coco_features(config):
    print("Starting feature extraction...")
    
    # Initialize feature extractor
    extractor = HierarchicalFeatureExtractor(device='cuda')
    extractor.model.eval()
    
    # Setup dataset
    dataset = CocoDatasetWrapper(
        root=str(config.coco_root / config.coco_train),
        annFile=str(config.ann_file),
        transform=extractor.transform
    )
    print(f"Found {len(dataset)} images")
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    # Extract features
    config.feature_save_dir.mkdir(parents=True, exist_ok=True)
    
    for batch_idx, (images, img_ids) in enumerate(tqdm(loader)):
        try:
            if images is None:
                continue
                
            # Extract features
            images = images.cuda()
            with torch.no_grad():
                features = extractor.extract_batch_features(images)
                
            # Save features for each image
            for idx, img_id in enumerate(img_ids):
                save_path = config.feature_save_dir / f"img_{img_id}.pt"
                image_features = [feat[idx:idx+1].cpu() for feat in features]
                torch.save(image_features, save_path)
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
            
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * loader.batch_size} images")
            
    print("Feature extraction complete!")


import torch
from PIL import Image
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import os
from .feature_extractor import HierarchicalFeatureExtractor

class COCODatasetWrapper(CocoDetection):
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Get correct image path
        img_path = os.path.join(self.root, img_info['file_name'])
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, img_id
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None, None

def custom_collate(batch):
    """Handle None values and create proper batches"""
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None
    imgs, ids = zip(*batch)
    return torch.stack(imgs), list(ids)
# def prepare_coco_features(config):
#     """Extract features from COCO dataset with optimized batch processing"""
#     print("Starting COCO feature extraction...")
    
#     # Initialize feature extractor
#     extractor = HierarchicalFeatureExtractor(device='cuda')
#     extractor.model.eval()
    
#     # Setup dataset
#     dataset = CocoDatasetWrapper(
#         root=str(config.coco_root / config.coco_train),
#         annFile=str(config.ann_file),
#         transform=extractor.transform
#     )
    
#     # Larger batch size and fewer workers for better GPU utilization
#     loader = DataLoader(
#         dataset,
#         batch_size=64,  # Increased batch size
#         num_workers=4,  # Reduced workers
#         pin_memory=True
#     )
    
#     # Process batches with caching
#     for batch_idx, (images, img_ids) in enumerate(tqdm(loader)):
#         try:
#             if images is None:
#                 continue
                
#             # Process batch
#             images = images.cuda(non_blocking=True)  # Use non_blocking transfer
#             features = extractor.extract_batch_features(images)
            
#             # Save features in batch
#             torch.cuda.synchronize()  # Ensure CUDA operations are complete
#             for idx, img_id in enumerate(img_ids):
#                 save_path = config.feature_save_dir / f"img_{img_id}.pt"
#                 image_features = [feat[idx:idx+1] for feat in features]
#                 torch.save(image_features, save_path)
                
#         except Exception as e:
#             print(f"Error processing batch {batch_idx}: {str(e)}")
#             continue
            
#         # Clear cache periodically
#         if (batch_idx + 1) % 10 == 0:
#             torch.cuda.empty_cache()
#             print(f"Processed {(batch_idx + 1) * loader.batch_size} images")
            
# def prepare_coco_features(config):
#     """Extract features from COCO dataset"""
#     print("Starting feature extraction...")
    
#     # Initialize feature extractor
#     extractor = HierarchicalFeatureExtractor(device='cuda')
#     extractor.model.eval()
    
#     # Setup dataset
#     dataset = COCODatasetWrapper(
#         root=str(config.coco_root / config.coco_train),
#         annFile=str(config.ann_file),
#         transform=extractor.transform
#     )
    
#     # Create dataloader
#     loader = DataLoader(
#         dataset,
#         batch_size=32,
#         num_workers=min(8, multiprocessing.cpu_count()),
#         collate_fn=custom_collate,
#         pin_memory=True
#     )
    
#     # Extract features
#     for batch_idx, (images, img_ids) in enumerate(tqdm(loader)):
#         try:
#             if images is None:
#                 continue
                
#             # Extract features
#             images = images.cuda()
#             with torch.no_grad():
#                 features = extractor.extract_batch_features(images)
                
#             # Save features for each image
#             for idx, img_id in enumerate(img_ids):
#                 save_path = config.feature_save_dir / f"img_{img_id}.pt"
#                 image_features = [feat[idx:idx+1] for feat in features]
#                 torch.save(image_features, save_path)
                
#         except Exception as e:
#             print(f"Error processing batch {batch_idx}: {str(e)}")
#             continue
            
#         if (batch_idx + 1) % 100 == 0:
#             print(f"Processed {(batch_idx + 1) * loader.batch_size} images")


# import torch
# from PIL import Image
# from torchvision.datasets import CocoDetection
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import multiprocessing
# from pathlib import Path
# import os
# from .feature_extractor import HierarchicalFeatureExtractor

# class COCODatasetWrapper(CocoDetection):
#     def __getitem__(self, idx):
#         coco = self.coco
#         img_id = self.ids[idx]
        
#         # Get image info and load image
#         img_info = coco.loadImgs(img_id)[0]
#         path = os.path.join(self.root, img_info['file_name'])
#         image = Image.open(path).convert('RGB')
        
#         # Apply transforms
#         if self.transform is not None:
#             try:
#                 image = self.transform(image)
#             except Exception as e:
#                 print(f"Error transforming image {path}: {str(e)}")
#                 return None
            
#         return image, img_id

# def custom_collate(batch):
#     """Remove None values and stack valid images"""
#     batch = [b for b in batch if b[0] is not None]
#     if not batch:
#         return None, None
#     images, ids = zip(*batch)
#     return torch.stack(images), list(ids)

# def prepare_coco_features(config):
#     """Extract features from COCO dataset"""
#     print("Starting feature extraction...")
    
#     # Initialize feature extractor
#     extractor = HierarchicalFeatureExtractor(device='cuda')
#     extractor.model.eval()
    
#     # Setup dataset and loader
#     dataset = COCODatasetWrapper(
#         root=str(config.coco_root / config.coco_train),
#         annFile=str(config.ann_file),
#         transform=extractor.transform
#     )
    
#     loader = DataLoader(
#         dataset,
#         batch_size=32,
#         num_workers=min(8, multiprocessing.cpu_count()),
#         collate_fn=custom_collate,
#         pin_memory=True
#     )
    
#     # Extract features
#     for batch_idx, (images, img_ids) in enumerate(tqdm(loader, desc="Extracting features")):
#         try:
#             if images is None:
#                 continue
                
#             # Process batch
#             images = images.cuda()
#             with torch.no_grad():
#                 features = extractor.extract_batch_features(images)
                
#             # Save features for each image
#             for idx, img_id in enumerate(img_ids):
#                 feat_list = [feat[idx:idx+1] for feat in features]
#                 save_path = config.feature_save_dir / f"img_{img_id}.pt"
#                 torch.save(feat_list, save_path)
                
#         except Exception as e:
#             print(f"Error processing batch {batch_idx}: {str(e)}")
#             continue
            
#         if (batch_idx + 1) % 100 == 0:
#             print(f"Processed {(batch_idx + 1) * loader.batch_size} images")

#     print("Feature extraction complete!")

# import torch
# from PIL import Image
# from torchvision.datasets import CocoDetection
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import multiprocessing
# from pathlib import Path
# from .feature_extractor import HierarchicalFeatureExtractor

# def custom_collate(batch):
#     """Custom collate function to handle batching"""
#     images = torch.stack([item[0] for item in batch])
#     targets = [item[1] for item in batch]
#     return images, targets

# class COCODatasetWrapper(CocoDetection):
#     def __getitem__(self, idx):
#         coco = self.coco
#         img_id = self.ids[idx]
        
#         # Load image
#         img_info = coco.loadImgs(img_id)[0]
#         image = Image.open(Path(self.root) / img_info['file_name']).convert('RGB')
        
#         # Get annotations
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         anns = coco.loadAnns(ann_ids)
        
#         # Apply transforms
#         if self.transform is not None:
#             image = self.transform(image)
            
#         return image, anns

# def prepare_coco_features(config):
#     # Initialize feature extractor
#     extractor = HierarchicalFeatureExtractor(device='cuda')
    
#     # Setup dataset
#     dataset = COCODatasetWrapper(
#         root=str(config.coco_root / config.coco_train),
#         annFile=str(config.ann_file),
#         transform=extractor.transform
#     )
    
#     # Create data loader
#     loader = DataLoader(
#         dataset,
#         batch_size=32,
#         num_workers=min(8, multiprocessing.cpu_count()),
#         collate_fn=custom_collate,
#         pin_memory=True
#     )
    
#     # Extract features
#     config.feature_save_dir.mkdir(parents=True, exist_ok=True)
    
#     for batch_idx, (images, _) in enumerate(tqdm(loader, desc="Extracting features")):
#         try:
#             # Process batch
#             with torch.no_grad():
#                 images = images.cuda()
#                 features = extractor.extract_batch_features(images)
                
#                 # Save individual image features
#                 for idx in range(images.size(0)):
#                     save_path = config.feature_save_dir / f"img_{batch_idx}_{idx}.pt"
#                     image_features = [feat[idx:idx+1] for feat in features]
#                     torch.save(image_features, save_path)
                    
#         except Exception as e:
#             print(f"Error processing batch {batch_idx}: {str(e)}")
#             continue
            
#         if (batch_idx + 1) % 100 == 0:
#             print(f"Processed {(batch_idx + 1) * loader.batch_size} images")
            
#     print("Feature extraction complete!")

# import torch
# from PIL import Image
# from torchvision.datasets import CocoDetection
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import multiprocessing
# from pathlib import Path
# import os
# from .feature_extractor import HierarchicalFeatureExtractor

# class CocoDatasetWrapper(CocoDetection):
#     def __getitem__(self, idx):
#         img_id = self.ids[idx]
#         img_path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
#         image = Image.open(img_path).convert('RGB')
#         target = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        
#         if self.transform is not None:
#             image = self.transform(image)
#         return image, target

# def prepare_coco_features(config):
#     print("Starting COCO feature extraction...")
    
#     # Initialize feature extractor
#     extractor = HierarchicalFeatureExtractor(device='cuda')
#     extractor.model.eval()
    
#     # Setup dataset
#     dataset = CocoDatasetWrapper(
#         root=str(config.coco_root / config.coco_train),
#         annFile=str(config.ann_file),
#         transform=extractor.transform
#     )
    
#     # Create data loader
#     loader = DataLoader(
#         dataset,
#         batch_size=32,
#         num_workers=min(8, multiprocessing.cpu_count()),
#         pin_memory=True
#     )
    
#     # Extract features
#     for batch_idx, (images, _) in enumerate(tqdm(loader)):
#         try:
#             with torch.no_grad():
#                 images = images.cuda()
#                 features = extractor.extract_batch_features(images)
                
#                 # Save features for each image
#                 for idx in range(images.size(0)):
#                     save_path = config.feature_save_dir / f"img_{batch_idx}_{idx}.pt"
#                     image_features = [feat[idx:idx+1] for feat in features]
#                     torch.save(image_features, save_path)
                    
#         except Exception as e:
#             print(f"Error processing batch {batch_idx}: {str(e)}")
#             continue
            
#         if (batch_idx + 1) % 100 == 0:
#             print(f"Processed {(batch_idx + 1) * loader.batch_size} images")
    
#     print("Feature extraction complete!")

# import torch
# from PIL import Image
# from torchvision.datasets import CocoDetection
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import multiprocessing
# from pathlib import Path
# import os

# class CocoDatasetWrapper(CocoDetection):
#     """Wrapper around CocoDetection to handle transforms properly"""
#     def __getitem__(self, idx):
#         # Get image ID and load image properly
#         img_id = self.ids[idx]
#         img_path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
#         image = Image.open(img_path).convert('RGB')
        
#         # Get annotations
#         target = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        
#         # Apply transforms
#         if self.transform is not None:
#             image = self.transform(image)
            
#         return image, target

# def prepare_coco_features(config):
#     """Extract features from COCO dataset"""
#     print("Starting COCO feature extraction...")
    
#     # Initialize feature extractor
#     extractor = HierarchicalFeatureExtractor(device='cuda')
#     extractor.model.eval()
    
#     # Setup dataset
#     dataset = CocoDatasetWrapper(
#         root=str(config.coco_root / config.coco_train),
#         annFile=str(config.ann_file),
#         transform=extractor.transform
#     )
#     print(f"Found {len(dataset)} images")
    
#     # Create data loader
#     loader = DataLoader(
#         dataset,
#         batch_size=32,
#         num_workers=min(8, multiprocessing.cpu_count()),
#         pin_memory=True
#     )
    
#     # Extract features
#     for batch_idx, (images, _) in enumerate(tqdm(loader)):
#         try:
#             with torch.no_grad():
#                 images = images.cuda()
#                 features = extractor.extract_batch_features(images)
                
#                 # Save features for each image
#                 for idx in range(images.size(0)):
#                     save_path = config.feature_save_dir / f"img_{batch_idx}_{idx}.pt"
#                     image_features = [feat[idx:idx+1] for feat in features]
#                     torch.save(image_features, save_path)
                    
#         except Exception as e:
#             print(f"Error processing batch {batch_idx}: {str(e)}")
#             continue
            
#         if (batch_idx + 1) % 100 == 0:
#             print(f"Processed {(batch_idx + 1) * loader.batch_size} images")
    
#     print("Feature extraction complete!")


