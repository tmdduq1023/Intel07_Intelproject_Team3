
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
        
        image_id = self.ids[idx]
        image = F.to_tensor(img)

        formatted_target = {}
        
        if not isinstance(target, list):
            target = [target]

        
        target = [obj for obj in target if 'bbox' in obj]

        boxes = torch.as_tensor([obj['bbox'] for obj in target], dtype=torch.float32)
        
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        formatted_target["boxes"] = boxes
        
        formatted_target["labels"] = torch.as_tensor([obj['category_id'] for obj in target], dtype=torch.int64)
        formatted_target["image_id"] = torch.tensor([image_id])
        formatted_target["area"] = torch.as_tensor([obj['area'] for obj in target], dtype=torch.float32)
        formatted_target["iscrowd"] = torch.as_tensor([obj['iscrowd'] for obj in target], dtype=torch.int64)

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
