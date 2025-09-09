
import torch
import torchvision
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def collate_fn(batch):
    return tuple(zip(*batch))

def _get_hw():
    import os
    h = int(os.getenv("AUG_RESIZE_H", "320"))
    w = int(os.getenv("AUG_RESIZE_W", "320"))
    return h, w

def get_train_transform():
    H, W = _get_hw()
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        # 추가 증강 (약하게 기본 포함)
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, cval=0, p=0.3),
        A.CLAHE(p=0.1),
        A.Resize(height=H, width=W),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

def get_val_transform():
    H, W = _get_hw()
    return A.Compose([
        A.Resize(height=H, width=W),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

from collections import defaultdict

class TransformedCocoDetection(CocoDetection):
    """A wrapper for CocoDetection to apply transforms to both image and target."""
    def __init__(self, root, annFile, train=False, target_rois=None):
        super(TransformedCocoDetection, self).__init__(root, annFile)
        self.train = train
        if self.train:
            self.transform = get_train_transform()
        else:
            self.transform = get_val_transform()

        if target_rois:
            print(f"[Dataset] Filtering for {len(target_rois)} target ROIs: {target_rois}")
            
            target_cat_ids = set()
            for cat in self.coco.dataset['categories']:
                if cat['name'] in target_rois:
                    target_cat_ids.add(cat['id'])
            
            if not target_cat_ids:
                print(f"Warning: None of the target ROIs {target_rois} found in dataset categories.")
                return

            # 1. Filter annotations
            filtered_anns = [ann for ann in self.coco.dataset['annotations'] if ann['category_id'] in target_cat_ids]
            
            # 2. Remap category IDs to a new contiguous range (1, 2, 3, ...)
            print("[Dataset] Remapping category IDs...")
            kept_cat_ids = sorted(list(target_cat_ids))
            id_map = {old_id: new_id for new_id, old_id in enumerate(kept_cat_ids, 1)}
            
            for ann in filtered_anns:
                ann['category_id'] = id_map[ann['category_id']]

            # 3. Update the categories list in the coco object as well
            new_categories = []
            for cat in self.coco.dataset['categories']:
                if cat['id'] in id_map:
                    cat['id'] = id_map[cat['id']]
                    new_categories.append(cat)
            self.coco.dataset['categories'] = sorted(new_categories, key=lambda x: x['id'])

            # 4. Filter image list and update coco object with remapped annotations
            image_ids_with_target_anns = set(ann['image_id'] for ann in filtered_anns)
            self.ids = sorted([img_id for img_id in self.ids if img_id in image_ids_with_target_anns])
            
            self.coco.dataset['annotations'] = filtered_anns
            self.coco.anns = {ann['id']: ann for ann in filtered_anns}
            self.coco.imgToAnns = defaultdict(list)
            for ann in filtered_anns:
                self.coco.imgToAnns[ann['image_id']].append(ann)

    def __getitem__(self, idx):
        img, target = super(TransformedCocoDetection, self).__getitem__(idx)

        W, H = img.size
        image = np.array(img)
        image_id = self.ids[idx]
        
        # Filter out annotations without valid bboxes
        target = [obj for obj in target if 'bbox' in obj and obj['bbox'][2] > 0 and obj['bbox'][3] > 0]

        bboxes = []
        category_ids = []
        if target:
            for obj in target:
                x, y, w, h = obj['bbox']
                # Clip the box to image boundaries to prevent errors
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, W)
                y2 = min(y + h, H)

                new_w = x2 - x1
                new_h = y2 - y1
                if new_w > 0 and new_h > 0:
                    bboxes.append([x1, y1, new_w, new_h])
                    category_ids.append(obj['category_id'])

        # Apply augmentations
        transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
        
        image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_category_ids = transformed['category_ids']

        formatted_target = {}
        
        if not transformed_bboxes:
            return image, {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty(0, dtype=torch.int64),
                "image_id": torch.tensor([image_id]),
                "area": torch.empty(0, dtype=torch.float32),
                "iscrowd": torch.empty(0, dtype=torch.int64)
            }

        # Convert bboxes from COCO [x, y, w, h] to Pascal VOC [x1, y1, x2, y2]
        boxes = torch.as_tensor(transformed_bboxes, dtype=torch.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        formatted_target["boxes"] = boxes
        
        labels = torch.as_tensor(transformed_category_ids, dtype=torch.int64)
        formatted_target["labels"] = labels
        
        formatted_target["image_id"] = torch.tensor([image_id])
        
        # Calculate area from the transformed bboxes
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        formatted_target["area"] = area
        
        # iscrowd is not handled by albumentations, we can assume 0 for simplicity
        iscrowd = torch.zeros((len(transformed_bboxes),), dtype=torch.int64)
        formatted_target["iscrowd"] = iscrowd

        return image, formatted_target

def get_fasterrcnn_model(num_classes):
    """Loads a pre-trained Faster R-CNN model and modifies the classifier head."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_ssdlite_model(num_classes):
    """Loads an SSDLite MobileNetV3 model with a custom classifier head."""
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None, 
        weights_backbone="DEFAULT", 
        num_classes=num_classes
    )
    return model

def get_retinanet_model(num_classes):
    """RetinaNet model (Focal Loss 내장)."""
    # Use pretrained backbone, fresh heads for our classes
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=None, 
        weights_backbone="DEFAULT",
        num_classes=num_classes
    )
    return model