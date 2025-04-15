from enum import Enum
import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50, fcn_resnet50, lraspp_mobilenet_v3_large,
    DeepLabV3_ResNet50_Weights, FCN_ResNet50_Weights, LRASPP_MobileNet_V3_Large_Weights
)
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F

try:
    from transformers import (
        Mask2FormerForUniversalSegmentation,
        OneFormerForUniversalSegmentation,
        AutoImageProcessor
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class SegmentationModel(Enum):
    DEEPLABV3 = "deeplabv3"
    FCN = "fcn"
    LRASPP = "lraspp"
    MASK2FORMER = "mask2former"
    ONEFORMER = "oneformer"

class SegmentationWrapper(nn.Module):
    def __init__(self, model_type: str, num_classes: int = None, pretrained: bool = True, device: str = None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = SegmentationModel(model_type.lower())
        self.num_classes = num_classes
        
        if self.model_type == SegmentationModel.DEEPLABV3:
            weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            self.model = deeplabv3_resnet50(weights=weights, num_classes=num_classes)
            self.processor = None
        
        elif self.model_type == SegmentationModel.FCN:
            weights = FCN_ResNet50_Weights.DEFAULT if pretrained else None
            self.model = fcn_resnet50(weights=weights, num_classes=num_classes)
            self.processor = None
            
        elif self.model_type == SegmentationModel.LRASPP:
            weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.model = lraspp_mobilenet_v3_large(weights=weights, num_classes=num_classes)
            self.processor = None
            
        elif HF_AVAILABLE and self.model_type == SegmentationModel.MASK2FORMER:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-ade-semantic")
            self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic", 
                                                             do_rescale=False)
            
        elif HF_AVAILABLE and self.model_type == SegmentationModel.ONEFORMER:
            model_name = "shi-labs/oneformer_ade20k_swin_tiny"
            self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name, do_rescale=False)
            self.task = "semantic"
            # Pre-compute task input indices
            self.task_inputs = torch.tensor([1] * 77, device=self.device)  # Default task token length is 77
        
        else:
            raise ValueError(f"Model type {model_type} not supported or HuggingFace not installed")
        
        # Add normalization transforms
        if self.processor is None:
            # For torchvision models
            self.transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        else:
            # For HuggingFace models
            self.transform = T.Compose([
                T.Lambda(lambda x: (x + 1) / 2),  # Convert from [-1, 1] to [0, 1]
                T.Lambda(lambda x: torch.clamp(x, 0, 1))  # Ensure values are in [0, 1]
            ])
        
        # Move model to device
        self.to(self.device)
        # Set model to evaluation mode
        self.eval()

    def _process_oneformer_output(self, outputs, input_shape):
        """Convert OneFormer output to standard segmentation format."""
        H, W = input_shape[-2:]
        
        # Get predictions from the logits
        logits = outputs.logits  # [B, H', W', C]
        
        # Move class dimension to proper position and interpolate if needed
        logits = logits.permute(0, 3, 1, 2)  # [B, C, H', W']
        
        # Interpolate to match input size if needed
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(
                logits,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        return logits

    @torch.no_grad()
    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Apply appropriate normalization
        x = self.transform(x)
        
        if self.processor is not None:
            # For HuggingFace models
            # Convert tensor to numpy for processor
            if torch.is_tensor(x):
                x = x.cpu().numpy()
                # Convert from BCHW to BHWC
                x = x.transpose(0, 2, 3, 1)
            
            # Process images with specific handling for OneFormer
            if self.model_type == SegmentationModel.ONEFORMER:
                batch_size = x.shape[0] if isinstance(x, np.ndarray) else len(x)
                # Process images
                inputs = self.processor(images=x, return_tensors="pt")
                # Add task inputs manually with correct shape
                inputs['task_inputs'] = self.task_inputs.unsqueeze(0).repeat(batch_size, 1)
            else:
                inputs = self.processor(images=x, return_tensors="pt")
                
            # Move processed inputs to device
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            
            # Forward pass with model-specific output handling
            outputs = self.model(**inputs)
            if self.model_type == SegmentationModel.ONEFORMER:
                return self._process_oneformer_output(outputs, x.shape)
            elif self.model_type == SegmentationModel.MASK2FORMER:
                return outputs.segmentation_logits
            else:
                return outputs.logits
        else:
            # For torchvision models
            return self.model(x)["out"]

    @property
    def supported_models(self):
        return [model.value for model in SegmentationModel]

# Usage example:
if __name__ == "__main__":
    # Create model
    model = SegmentationWrapper("oneformer", num_classes=21)
    
    # Create normalized dummy input (values roughly in [-1, 1])
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).clamp(-1, 1)
    
    # Run inference
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Model type: {model.model_type.value}")
        print(f"Output shape: {output.shape}")
        print(f"Output device: {output.device}")
