import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import timm
import torch.nn as nn
import torchvision
from collections import defaultdict

# --- STAGE 1: ROI Detection (from infer_roi_v2.py and utils_v3.py) ---
try:
    from utils_v3 import get_retinanet_model
except ImportError:
    print("Error: Could not import from utils_v3.py. Please ensure it is in the same directory.")
    sys.exit(1)

# --- STAGE 2: Skin Feature Analysis (from train_skin_roi_v2_1.py) ---
class RoiMultiHead(nn.Module):
    def __init__(self, backbone="resnet50", roi_label_space=None, pretrained=True):
        super().__init__()
        assert roi_label_space is not None
        self.roi_reg_keys = roi_label_space["regression_keys"]
        self.roi_cls_keys = roi_label_space["class_keys"]
        self.roi_cls_maps = roi_label_space["class_key_to_index"]
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        in_dim = self.backbone.num_features
        self.reg_heads = nn.ModuleDict()
        self.cls_heads = nn.ModuleDict()
        for roi, R_keys in self.roi_reg_keys.items():
            if len(R_keys) > 0:
                self.reg_heads[roi] = nn.Linear(in_dim, len(R_keys))
        for roi, C_keys in self.roi_cls_keys.items():
            if len(C_keys) > 0:
                heads = nn.ModuleList()
                for ck in C_keys:
                    nclass = len(self.roi_cls_maps.get(roi, {}).get(ck, {}).get("values", []))
                    heads.append(nn.Linear(in_dim, nclass))
                self.cls_heads[roi] = heads

    def forward(self, x, roi_names):
        feats = self.backbone(x)
        B = feats.size(0)
        out_reg, out_cls = [None]*B, [None]*B
        for i in range(B):
            roi = roi_names[i]
            f = feats[i:i+1]
            reg_pred = None
            if roi in self.reg_heads:
                reg_pred = self.reg_heads[roi](f)
            cls_preds = None
            if roi in self.cls_heads:
                cls_preds = [head(f) for head in self.cls_heads[roi]]
            out_reg[i], out_cls[i] = reg_pred, cls_preds
        return out_reg, out_cls

# --- Configuration ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- STAGE 1 Config ---
DETECTOR_MODEL_PATH = 'best_roi_detector_retinanet_map.pth'
DETECTOR_TARGET_ROIS = ['facepart::forehead', 'facepart::lips', 'facepart::left_cheek', 'facepart::right_cheek', 'facepart::chin']
DETECTOR_NUM_CLASSES = len(DETECTOR_TARGET_ROIS) + 1
DETECTOR_LABEL_MAP_PATH = '../dataset/roi_coco_training.json' # Relative path for remapping

# --- STAGE 2 Config ---
SKIN_MODEL_PATH = 'runs/skin_roi_v4/best.pth'
SKIN_REG_STD_PATH = 'runs/skin_roi_v4/reg_std.json'

# --- Helper Functions ---
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# --- STAGE 1 INFERENCE FUNCTIONS ---
def load_detector_model(num_classes, model_path):
    model = get_retinanet_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def get_detector_label_map(coco_path, target_rois):
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)
    target_cat_ids = {cat['id'] for cat in coco_data['categories'] if cat['name'] in target_rois}
    original_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    # This map is new_id -> old_id
    id_map = {new_id: old_id for new_id, old_id in enumerate(sorted(list(target_cat_ids)), 1)}
    # The return map should be new_id -> name
    return {new_id: original_id_to_name[old_id] for new_id, old_id in id_map.items()}

def detect_rois(detector_model, label_map, image_path):
    image = Image.open(image_path).convert("RGB")
    transform = A.Compose([A.Resize(320, 320), A.ToFloat(max_value=255.0), ToTensorV2()])
    image_tensor = transform(image=np.array(image))['image'].unsqueeze(0)
    
    with torch.no_grad():
        predictions = detector_model(image_tensor.to(DEVICE))[0]

    boxes, labels, scores = predictions['boxes'].cpu().numpy(), predictions['labels'].cpu().numpy(), predictions['scores'].cpu().numpy()
    print(f"DEBUG: Raw predicted label IDs from model: {labels}")
    
    best_detections = {}
    for i in range(len(boxes)):
        label_id = labels[i]
        if label_id not in best_detections or scores[i] > best_detections[label_id]['score']:
            best_detections[label_id] = {'box': boxes[i], 'score': scores[i]}

    detected_rois = []
    img_width, img_height = image.size
    scale_x, scale_y = img_width / 320.0, img_height / 320.0
    for label_id, det in best_detections.items():
        x1, y1, x2, y2 = det['box']
        detected_rois.append({
            'box': [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)],
            'label_name': label_map.get(label_id, 'Unknown')
        })
    return image, detected_rois

# --- STAGE 2 INFERENCE FUNCTIONS ---
def load_skin_model(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    model = RoiMultiHead(backbone='convnext_tiny', roi_label_space=ckpt['roi_label_space'])
    model.load_state_dict(ckpt['model'])
    model.to(DEVICE)
    model.eval()
    return model, ckpt['roi_label_space'], ckpt.get('reg_std')

def analyze_skin(skin_model, roi_label_space, reg_std, cropped_image, roi_name):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((384, 384)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(cropped_image).unsqueeze(0)

    with torch.no_grad():
        reg_pred, cls_pred = skin_model(image_tensor.to(DEVICE), [roi_name])

    results = {}
    # --- Process Regression Results ---
    reg_keys = roi_label_space['regression_keys'].get(roi_name, [])
    if reg_pred[0] is not None:
        reg_values = reg_pred[0].squeeze(0).cpu().numpy()
        for i, key in enumerate(reg_keys):
            val = reg_values[i]
            if reg_std and key in reg_std:
                mean, std = reg_std[key]['mean'], reg_std[key]['std']
                val = val * std + mean
            results[key] = f'{val:.2f}'

    # --- Process Classification Results ---
    cls_keys = roi_label_space['class_keys'].get(roi_name, [])
    cls_maps = roi_label_space['class_key_to_index'].get(roi_name, {})
    if cls_pred and cls_pred[0] is not None:
        for i, key in enumerate(cls_keys):
            logits = cls_pred[0][i]
            predicted_idx = logits.argmax(dim=1).item()
            
            idx_to_val_map = cls_maps.get(key, {}).get('idx_to_val', {})
            predicted_value = idx_to_val_map.get(predicted_idx, 'Error: Unknown Index')
            results[key] = predicted_value
            
    return results

# --- MAIN ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_image>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at {input_image_path}")
        sys.exit(1)

    print("--- Loading Models ---")
    detector_model = load_detector_model(DETECTOR_NUM_CLASSES, DETECTOR_MODEL_PATH)
    skin_model, skin_label_space, skin_reg_std = load_skin_model(SKIN_MODEL_PATH)
    
    print("\n--- Stage 1: Detecting Face ROIs ---")
    original_image, detected_rois = detect_rois(detector_model, get_detector_label_map(DETECTOR_LABEL_MAP_PATH, DETECTOR_TARGET_ROIS), input_image_path)
    
    if not detected_rois:
        print("No ROIs were detected in the image.")
        sys.exit(0)

    print(f"\nDetected {len(detected_rois)} ROIs: {[r['label_name'] for r in detected_rois]}")

    print("\n--- Stage 2: Analyzing Skin Features per ROI ---")
    final_results = defaultdict(dict)
    for roi_info in detected_rois:
        roi_name = roi_info['label_name'].split('::', 1)[1]
        x1, y1, x2, y2 = roi_info['box']
        cropped_roi = original_image.crop((x1, y1, x2, y2))
        
        analysis_results = analyze_skin(skin_model, skin_label_space, skin_reg_std, cropped_roi, roi_name)
        final_results[roi_name] = analysis_results
        
    print("\n--- FINAL RESULTS ---")
    print(json.dumps(final_results, indent=2, ensure_ascii=False))
