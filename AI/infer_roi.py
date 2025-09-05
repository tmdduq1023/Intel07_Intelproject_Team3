import torch
import torchvision
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import json
import os
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add the parent directory to the path to import utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from utils import get_fasterrcnn_model # Assuming get_fasterrcnn_model is in utils.py

# Define the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- Configuration ---
MODEL_PATH = 'best_roi_detector_res50.pth' # Path to the trained model
LABEL_MAP_PATH = '../dataset/roi_coco_training.json' # Path to the COCO annotation file for label mapping
OUTPUT_DIR = 'detected_rois' # Directory to save cropped ROIs

# We have 9 ROI categories + 1 background class
NUM_CLASSES = 10 


def load_model(num_classes, model_path):
    """Loads the Faster R-CNN EfficientNet model and its trained weights."""
    model = get_fasterrcnn_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    return model

def load_label_map(coco_annotation_path):
    """Loads the label mapping from the COCO annotation file."""
    with open(coco_annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    label_map = {}
    # Category ID 0 is typically background, so we start from 1 or map based on actual IDs
    # COCO format category IDs can start from 1 or be arbitrary.
    # We need to ensure all categories are mapped.
    for category in coco_data['categories']:
        label_map[category['id']] = category['name']
    return label_map

def preprocess_image(image_path):
    """
    Preprocesses a single image for inference.
    Matches the validation preprocessing from utils.py.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Define the same transforms as in utils.py for validation
    transform = A.Compose([
        A.Resize(320, 320),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image_np)
    return transformed['image'].unsqueeze(0), image # Return tensor and original PIL image

def perform_inference(model, image_tensor):
    """Performs inference on the preprocessed image tensor."""
    with torch.no_grad():
        predictions = model(image_tensor.to(DEVICE))[0] # Get predictions for the first (and only) image in batch
    return predictions

def process_detections(predictions, label_map, original_image_size):
    """
    Filters predictions to select the highest scoring detection for each class,
    and maps labels to names. Returns a list of dictionaries, each representing a detected ROI.
    """
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    best_detections_per_class = {}

    for i in range(len(boxes)):
        box = boxes[i]
        label_id = labels[i]
        score = scores[i]

        # Assuming label_id 0 is background and we are interested in other ROI classes
        # NUM_CLASSES is 10, so label_ids 1-9 are ROI categories
        if label_id != 0 and label_id < NUM_CLASSES: # Ensure it's an ROI class
            if label_id not in best_detections_per_class or score > best_detections_per_class[label_id]['score']:
                best_detections_per_class[label_id] = {
                    'box': box,
                    'label_id': label_id,
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

        # Ensure coordinates are within image bounds
        x1_orig = max(0, x1_orig)
        y1_orig = max(0, y1_orig)
        x2_orig = min(img_width, x2_orig)
        y2_orig = min(img_height, y2_orig)

        detected_rois.append({
            'box': [x1_orig, y1_orig, x2_orig, y2_orig],
            'label_id': label_id,
            'label_name': label_map.get(label_id, 'Unknown'),
            'score': score
        })
    return detected_rois

def save_rois(original_image, detected_rois, output_dir, image_name):
    """Crops and saves detected ROIs from the original image."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Draw bounding boxes on a copy of the original image for visualization
    image_with_boxes = original_image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    try:
        font = ImageFont.truetype("arial.ttf", 15) # Try to use a common font
    except IOError:
        font = ImageFont.load_default() # Fallback to default

    for i, roi in enumerate(detected_rois):
        x1, y1, x2, y2 = roi['box']
        label_name = roi['label_name']
        score = roi['score']

        # Crop the ROI
        cropped_roi = original_image.crop((x1, y1, x2, y2))
        roi_filename = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_{label_name}_{i}.png")
        cropped_roi.save(roi_filename)
        print(f"Saved {label_name} to {roi_filename}")

        # Draw box and label on the visualization image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{label_name}: {score:.2f}"
        # Draw text with a background for better visibility
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
        draw.text((x1 + 2, y1 - text_height - 2), text, fill="white", font=font)

    # Save the image with drawn bounding boxes
    image_with_boxes_filename = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_detections.png")
    image_with_boxes.save(image_with_boxes_filename)
    print(f"Saved detections visualization to {image_with_boxes_filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer_roi.py <path_to_image>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    image_name = os.path.basename(input_image_path)

    # Load model and label map
    model = load_model(NUM_CLASSES, MODEL_PATH)
    label_map = load_label_map(LABEL_MAP_PATH)

    # Preprocess image
    image_tensor, original_image_pil = preprocess_image(input_image_path)
    
    # Perform inference
    predictions = perform_inference(model, image_tensor)

    # Process detections
    detected_rois = process_detections(predictions, label_map, original_image_pil.size)

    # Save ROIs and visualization
    save_rois(original_image_pil, detected_rois, OUTPUT_DIR, image_name)

    print("Inference complete.")