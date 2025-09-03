
import os
import json
import glob
from tqdm import tqdm

def convert_to_coco(dataset_path, output_file):
    """
    Converts a custom dataset format to COCO format.

    The script expects a directory structure like:
    dataset_path/
    ├── Training/
    │   ├── labeled_data/
    │   │   ├── 1_digital_camera/
    │   │   │   ├── 0002/
    │   │   │   │   ├── 0002_01_F_00.json
    │   │   │   │   └── ...
    │   │   └── ...
    │   └── origin_data/
    │       ├── 1_digital_camera/
    │       │   ├── 0002/
    │       │   │   ├── 0002_01_F.jpg
    │       │   │   └── ...
    │       └── ...
    └── ...

    Args:
        dataset_path (str): The root path of the dataset.
        output_file (str): The path to save the output COCO annotation file.
    """
    coco_format = {
        "info": {
            "description": "Custom Dataset to COCO format",
            "version": "1.0",
            "year": 2025,
            "contributor": "Bbang",
            "date_created": "2025/09/03"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_counter = 1
    annotation_id_counter = 1
    category_map = {}
    category_id_counter = 1

    training_path = os.path.join(dataset_path, 'Training')
    origin_data_path = os.path.join(training_path, 'origin_data')
    labeled_data_path = os.path.join(training_path, 'labeled_data')

    # Find all image files first
    image_files = glob.glob(os.path.join(origin_data_path, '**', '*.jpg'), recursive=True)
    
    print(f"Found {len(image_files)} images to process.")

    for image_path in tqdm(image_files, desc="Processing images"):
        # Extract relative path to construct labeled data path
        relative_image_path = os.path.relpath(image_path, origin_data_path)
        
        # Get image dimensions from the first corresponding JSON file
        json_base_path = os.path.join(labeled_data_path, os.path.splitext(relative_image_path)[0])
        
        # Find all corresponding json files
        label_files = glob.glob(f"{json_base_path}*.json")
        
        if not label_files:
            continue

        # Read the first json file to get image width and height
        try:
            width, height = -1, -1
            for lf in label_files:
                 with open(lf, 'r') as f:
                    temp_data = json.load(f)
                    if (temp_data.get('images') and
                        temp_data['images'].get('width') and
                        temp_data['images'].get('height') and
                        temp_data['images'].get('bbox')):
                        # Heuristic: find the bbox that covers the whole image to get dimensions
                        if temp_data['images']['bbox'][0] == 0 and temp_data['images']['bbox'][1] == 0:
                             width = temp_data['images']['width']
                             height = temp_data['images']['height']
                             break
            
            if width == -1 or height == -1:
                # Fallback if no full-image bbox is found in a clean file
                with open(label_files[0], 'r') as f:
                    first_label_data = json.load(f)
                    if first_label_data.get('images'):
                        width = first_label_data['images'].get('width', -1)
                        height = first_label_data['images'].get('height', -1)

            if width == -1 or height == -1:
                print(f"Warning: Could not determine dimensions for {image_path}. Skipping.")
                continue

        except (IOError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not read or parse base label file for {image_path}. Skipping. Error: {e}")
            continue

        # Add image info to COCO structure
        image_info = {
            "id": image_id_counter,
            "width": width,
            "height": height,
            "file_name": relative_image_path.replace(os.path.sep, '/')
        }
        coco_format["images"].append(image_info)

        # Process each label file for this image
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    label_data = json.load(f)

                annotations = label_data.get('annotations')
                if not annotations:
                    continue
                
                # Iterate through annotations in the current json file
                for category_name, value in annotations.items():
                    if value is not None and value != 0:
                        
                        if category_name not in category_map:
                            category_map[category_name] = category_id_counter
                            coco_format['categories'].append({
                                "id": category_id_counter,
                                "name": category_name,
                                "supercategory": "object"
                            })
                            category_id_counter += 1
                        
                        category_id = category_map[category_name]

                        bbox_data = label_data.get('images', {}).get('bbox')
                        if bbox_data:
                            x_min, y_min, x_max, y_max = bbox_data
                            width_bbox = x_max - x_min
                            height_bbox = y_max - y_min
                            
                            annotation_info = {
                                "id": annotation_id_counter,
                                "image_id": image_id_counter,
                                "category_id": category_id,
                                "bbox": [x_min, y_min, width_bbox, height_bbox],
                                "area": width_bbox * height_bbox,
                                "iscrowd": 0,
                                "segmentation": []
                            }
                            coco_format['annotations'].append(annotation_info)
                            annotation_id_counter += 1

            except (IOError, json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not process label file {label_file}. Skipping. Error: {e}")
                continue
        
        image_id_counter += 1

    # Save the final COCO JSON file
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)
    
    print(f"\nConversion complete. COCO annotations saved to {output_file}")
    print(f"Summary: {len(coco_format['images'])} images, {len(coco_format['annotations'])} annotations, {len(coco_format['categories'])} categories.")


if __name__ == '__main__':
    root_path = os.getcwd() 
    dataset_root = os.path.join(root_path, 'dataset')
    output_json_file = os.path.join(dataset_root, 'coco_annotations.json')
    
    convert_to_coco(dataset_root, output_json_file)
