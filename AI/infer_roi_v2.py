import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

# Assumes utils_v2.py is in the same directory or accessible
try:
    from utils_v3 import get_retinanet_model
except ImportError:
    print("Error: Could not import get_retinanet_model from utils_v2.py.")
    print("Please ensure utils_v2.py is in the same directory.")
    sys.exit(1)

# --- Configuration ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define the ROIs that the model was trained on
TARGET_ROIS = ['facepart::forehead', 'facepart::lips', 'facepart::left_cheek', 'facepart::right_cheek', 'facepart::chin']
NUM_CLASSES = len(TARGET_ROIS) + 1  # +1 for background

# Define paths relative to the project root
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path('.').resolve().parent

MODEL_PATH = PROJECT_ROOT / 'AI' / 'best_roi_detector_retinanet_map.pth'
LABEL_MAP_PATH = PROJECT_ROOT / 'dataset' / 'roi_coco_training.json'
OUTPUT_DIR = PROJECT_ROOT / 'AI' / 'detected_rois_v2'

def load_model(num_classes, model_path):
    """Loads the RetinaNet model and its trained weights."""
    if not Path(model_path).exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)
    model = get_retinanet_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_label_map(coco_annotation_path, target_rois):
    """
    Loads the label mapping from the COCO annotation file and remaps it
    to match the model's output.
    """
    if not Path(coco_annotation_path).exists():
        print(f"Error: COCO annotation file not found at {coco_annotation_path}")
        sys.exit(1)
        
    with open(coco_annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    target_cat_ids = set()
    original_id_to_name = {}
    for category in coco_data['categories']:
        original_id_to_name[category['id']] = category['name']
        if category['name'] in target_rois:
            target_cat_ids.add(category['id'])
            
    kept_cat_ids = sorted(list(target_cat_ids))
    id_map = {old_id: new_id for new_id, old_id in enumerate(kept_cat_ids, 1)}
    
    new_label_map = {new_id: original_id_to_name[old_id] for old_id, new_id in id_map.items()}
    print(f"Loaded and remapped labels: {new_label_map}")
    return new_label_map

def preprocess_image(image_path):
    """
    Preprocesses a single image for inference.
    Matches the validation preprocessing from utils_v2.py.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    transform = A.Compose([
        A.Resize(320, 320), # Assuming default size from training
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image_np)
    return transformed['image'].unsqueeze(0), image

def perform_inference(model, image_tensor):
    """Performs inference on the preprocessed image tensor."""
    with torch.no_grad():
        predictions = model(image_tensor.to(DEVICE))[0]
    return predictions

def process_detections(predictions, label_map, original_image_size):
    """
    Selects the highest scoring detection for each class, regardless of score.
    Returns a list of dictionaries, each representing a detected ROI.
    """
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    best_detections_per_class = {}

    for i in range(len(boxes)):
        box = boxes[i]
        label_id = labels[i]
        score = scores[i]

        # Find the highest scoring box for each class
        if label_id not in best_detections_per_class or score > best_detections_per_class[label_id]['score']:
            best_detections_per_class[label_id] = {
                'box': box,
                'score': score
            }

    detected_rois = []
    img_width, img_height = original_image_size

    for label_id, detection_info in best_detections_per_class.items():
        box = detection_info['box']
        score = detection_info['score']

        # Scale bounding box coordinates back to original image size
        scale_x = img_width / 320.0
        scale_y = img_height / 320.0

        x1, y1, x2, y2 = box
        x1_orig = int(x1 * scale_x)
        y1_orig = int(y1 * scale_y)
        x2_orig = int(x2 * scale_x)
        y2_orig = int(y2 * scale_y)

        detected_rois.append({
            'box': [x1_orig, y1_orig, x2_orig, y2_orig],
            'label_id': label_id,
            'label_name': label_map.get(label_id, 'Unknown'),
            'score': score
        })
    return detected_rois

def save_rois(original_image, detected_rois, output_dir, image_name):
    """Crops and saves detected ROIs and a visualization image."""
    os.makedirs(output_dir, exist_ok=True)
    
    image_with_boxes = original_image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for i, roi in enumerate(detected_rois):
        x1, y1, x2, y2 = roi['box']
        label_name = roi['label_name']
        score = roi['score']

        cropped_roi = original_image.crop((x1, y1, x2, y2))
        sanitized_label = label_name.replace("::", "_")
        roi_filename = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_{sanitized_label}_{i}.png")
        cropped_roi.save(roi_filename)
        print(f"Saved {label_name} to {roi_filename}")

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{label_name}: {score:.2f}"
        text_bbox = draw.textbbox((x1, y1 - 15), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x1, y1 - 15), text, fill="white", font=font)

    image_with_boxes_filename = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_detections.png")
    image_with_boxes.save(image_with_boxes_filename)
    print(f"Saved detections visualization to {image_with_boxes_filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_image>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at {input_image_path}")
        sys.exit(1)

    image_name = os.path.basename(input_image_path)

    model = load_model(NUM_CLASSES, MODEL_PATH)
    label_map = load_label_map(LABEL_MAP_PATH, TARGET_ROIS)
    image_tensor, original_image_pil = preprocess_image(input_image_path)
    predictions = perform_inference(model, image_tensor)
    detected_rois = process_detections(predictions, label_map, original_image_pil.size)
    save_rois(original_image_pil, detected_rois, OUTPUT_DIR, image_name)

    print("Inference complete.")
