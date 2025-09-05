import torch
import torchvision
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead


def collate_fn(batch):
    return tuple(zip(*batch))



class TransformedCocoDetection(CocoDetection):
    """A wrapper for CocoDetection to apply transforms to both image and target."""
    def __init__(self, root, annFile, train=False):
        super(TransformedCocoDetection, self).__init__(root, annFile)
        self.train = train

    def __getitem__(self, idx):
        img, target = super(TransformedCocoDetection, self).__getitem__(idx)
        
        # Get original image size for scaling bboxes
        original_w, original_h = img.size

        # Resize image to a fixed size (e.g., 320x320 for SSDLite320)
        # This is crucial for performance and memory usage.
        target_size = (320, 320)
        img = F.resize(img, target_size)
        image = F.to_tensor(img)

        # Prepare target dictionary
        image_id = self.ids[idx]
        formatted_target = {}
        
        if not isinstance(target, list):
            target = [target]
        
        target = [obj for obj in target if 'bbox' in obj and obj['bbox'][2] > 0 and obj['bbox'][3] > 0]

        if not target: # If no valid annotations, return empty tensors
            return image, {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty(0, dtype=torch.int64),
                "image_id": torch.tensor([image_id]),
                "area": torch.empty(0, dtype=torch.float32),
                "iscrowd": torch.empty(0, dtype=torch.int64)
            }

        boxes = torch.as_tensor([obj['bbox'] for obj in target], dtype=torch.float32)
        
        # Scale bounding boxes to match the resized image
        scale_w = target_size[0] / original_w
        scale_h = target_size[1] / original_h
        boxes[:, 0] *= scale_w
        boxes[:, 1] *= scale_h
        boxes[:, 2] *= scale_w
        boxes[:, 3] *= scale_h

        # Convert bbox format from [x, y, w, h] to [x1, y1, x2, y2]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        formatted_target["boxes"] = boxes
        
        labels = torch.as_tensor([obj['category_id'] for obj in target], dtype=torch.int64)
        formatted_target["labels"] = labels
        
        formatted_target["image_id"] = torch.tensor([image_id])
        
        area = torch.as_tensor([obj['area'] for obj in target], dtype=torch.float32)
        # Scale area as well
        formatted_target["area"] = area * (scale_w * scale_h)
        
        iscrowd = torch.as_tensor([obj['iscrowd'] for obj in target], dtype=torch.int64)
        formatted_target["iscrowd"] = iscrowd

        # Apply horizontal flip augmentation
        if self.train and torch.rand(1) < 0.5:
            image = F.hflip(image)
            img_width, _ = F.get_image_size(image)
            formatted_target["boxes"][:, [0, 2]] = img_width - formatted_target["boxes"][:, [2, 0]]

        return image, formatted_target

def get_fasterrcnn_model(num_classes):
    """
    Loads a pre-trained Faster R-CNN model and modifies the classifier head.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_ssdlite_model(num_classes):
    """
    Loads an SSDLite MobileNetV3 model with a custom classifier head.
    This is the correct way to instantiate a model for transfer learning.
    """
    
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None, 
        weights_backbone="DEFAULT", 
        num_classes=num_classes
    )
    return model
