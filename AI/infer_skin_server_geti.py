
import torch
from PIL import Image
import numpy as np
import json
import os
import sys
from pathlib import Path
import timm
import torch.nn as nn
import torchvision
from collections import defaultdict
import requests
import cv2

# --- SDK Imports ---
try:
    from geti_sdk.deployment import Deployment
except ImportError:
    print("Error: geti-sdk is not installed. Please run: pip install geti-sdk==2.6.*")
    sys.exit(1)

# --- STAGE 2: Skin Feature Analysis Model Definition ---
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

# --- Configurations ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SERVER_URL = "http://127.0.0.1:5000/receive"

# --- STAGE 1 Config (Geti) ---
GETI_DEPLOYMENT_PATH = 'geti_face_v2/deployment'

# --- STAGE 2 Config (Skin Analysis) ---
SKIN_MODEL_PATH = 'runs/skin_roi_v4/best.pth'

# --- STAGE 1: GETI ROI DETECTION ---
def load_geti_deployment(deployment_path):
    print("Loading Intel Geti deployment...")
    deployment = Deployment.from_folder(deployment_path)
    print("Loading inference models to device...")
    deployment.load_inference_models(device="CPU")
    return deployment

def detect_rois_with_geti(deployment, image_path):
    print(f"Reading image: {image_path}")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Inferring with Geti model...")
    prediction = deployment.infer(image_rgb)

    detected_rois = []
    img_height, img_width, _ = image_rgb.shape
    image_center_x = img_width / 2

    for annotation in prediction.annotations:
        shape = annotation.shape
        box = [int(shape.x), int(shape.y), int(shape.x + shape.width), int(shape.y + shape.height)]
        label_name = annotation.labels[0].name

        if label_name == 'cheek':
            box_center_x = box[0] + (box[2] - box[0]) / 2
            final_label = 'facepart::right_cheek' if box_center_x < image_center_x else 'facepart::left_cheek'
        elif label_name == 'lip':
             final_label = 'facepart::lips'
        else:
            final_label = f"facepart::{label_name}"

        detected_rois.append({
            'box': box,
            'label_name': final_label
        })
    return Image.fromarray(image_rgb), detected_rois

# --- STAGE 2: SKIN ANALYSIS ---
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
    reg_keys = roi_label_space['regression_keys'].get(roi_name, [])
    if reg_pred[0] is not None:
        reg_values = reg_pred[0].squeeze(0).cpu().numpy()
        for i, key in enumerate(reg_keys):
            val = reg_values[i]
            if reg_std and key in reg_std:
                mean, std = reg_std[key]['mean'], reg_std[key]['std']
                val = val * std + mean
            results[key] = float(f'{val:.2f}')
    cls_keys = roi_label_space['class_keys'].get(roi_name, [])
    cls_maps = roi_label_space['class_key_to_index'].get(roi_name, {})
    if cls_pred and cls_pred[0] is not None:
        for i, key in enumerate(cls_keys):
            logits = cls_pred[0][i]
            predicted_idx = logits.argmax(dim=1).item()
            idx_to_val_map = cls_maps.get(key, {}).get('idx_to_val', {})
            predicted_value = idx_to_val_map.get(str(predicted_idx), idx_to_val_map.get(predicted_idx, 'Error'))
            results[key] = predicted_value
    return results

# --- STAGE 3: SERVER SEND ---
def send_results_to_server(results, server_url):
    try:
        payload = {
            "forehead": results.get("forehead", {}),
            "l_check": results.get("left_cheek", {}),
            "r_check": results.get("right_cheek", {}),
            "chin": results.get("chin", {}),
            "lib": results.get("lips", {})
        }
        print("\n--- Sending to Server ---")
        print("Payload:", json.dumps(payload, indent=2))
        response = requests.post(server_url, json=payload, timeout=5)
        response.raise_for_status()
        print(f"Server Response ({response.status_code}): {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"\nError: Could not connect to the server at {server_url}. Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during data transformation or sending: {e}")

def send_detection_failure_to_server(missing_parts, server_url):
    """Sends a detection failure message to the server."""
    try:
        error_message = f"ROI detection failed: {', '.join(missing_parts)}"
        payload = {"error": error_message}
        
        print("\n--- Sending Detection Failure to Server ---")
        print("Payload:", json.dumps(payload, indent=2))
        
        response = requests.post(server_url, json=payload, timeout=5)
        response.raise_for_status()
        
        print(f"Server Response ({response.status_code}): {response.json()}")

    except requests.exceptions.RequestException as e:
        print(f"\nError: Could not connect to the server at {server_url}. Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred while sending failure report: {e}")


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
    geti_deployment = load_geti_deployment(GETI_DEPLOYMENT_PATH)
    skin_model, skin_label_space, skin_reg_std = load_skin_model(SKIN_MODEL_PATH)
    
    print("--- Stage 1: Detecting Face ROIs with Intel Geti ---")
    original_image, detected_rois = detect_rois_with_geti(geti_deployment, input_image_path)
    
    # --- Check for missing ROIs ---
    expected_rois = {'facepart::forehead', 'facepart::lips', 'facepart::left_cheek', 'facepart::right_cheek', 'facepart::chin'}
    detected_labels = {r['label_name'] for r in detected_rois}
    missing_rois = expected_rois - detected_labels

    if missing_rois:
        print(f"\nWarning: The following ROIs were not detected: {', '.join(missing_rois)}")
        send_detection_failure_to_server(list(missing_rois), SERVER_URL)
        sys.exit(0)

    print(f"\nDetected {len(detected_rois)} ROIs: {[r['label_name'] for r in detected_rois]}")

    print("--- Stage 2: Analyzing Skin Features per ROI ---")
    final_results = defaultdict(dict)
    for roi_info in detected_rois:
        if '::' not in roi_info['label_name']:
            continue
        roi_name = roi_info['label_name'].split('::', 1)[1]
        x1, y1, x2, y2 = roi_info['box']
        cropped_roi = original_image.crop((x1, y1, x2, y2))
        
        analysis_results = analyze_skin(skin_model, skin_label_space, skin_reg_std, cropped_roi, roi_name)
        final_results[roi_name] = analysis_results
        
    print("--- FINAL RESULTS ---")
    print(json.dumps(final_results, indent=2, ensure_ascii=False))

    # --- STAGE 3: 서버로 결과 전송 ---
    send_results_to_server(final_results, SERVER_URL)


