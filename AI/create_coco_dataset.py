
import json
import os
import glob

def create_coco_dataset():
    base_dir = "/home/bbang/Workspace/Intel07_Intelproject_Team3/dataset"
    output_file = "/home/bbang/Workspace/Intel07_Intelproject_Team3/coco_skin_dataset.json"
    
    coco_output = {
        "info": {
            "description": "Skin Analysis Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Gemini",
            "date_created": "2025/09/03"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_map = {}
    category_id_counter = 1

    def get_category_id(name):
        nonlocal category_id_counter
        if name not in category_map:
            category_map[name] = category_id_counter
            coco_output["categories"].append({
                "id": category_id_counter,
                "name": name,
                "supercategory": "skin_metrics"
            })
            category_id_counter += 1
        return category_map[name]

    image_id_counter = 1
    annotation_id_counter = 1

    def process_and_add_annotations(current_key_prefix, current_value, image_id):
        nonlocal annotation_id_counter
        if current_value is None:
            return

        if isinstance(current_value, (int, float)): # Numerical scalar value
            cat_id = get_category_id(current_key_prefix)
            annotation_entry = {
                "id": annotation_id_counter,
                "image_id": image_id,
                "category_id": cat_id,
                "value": current_value,
                "area": 0,
                "bbox": [],
                "iscrowd": 0
            }
            coco_output["annotations"].append(annotation_entry)
            annotation_id_counter += 1
        elif isinstance(current_value, str): # String value
            try:
                # Try to convert string to float
                float_value = float(current_value)
                cat_id = get_category_id(current_key_prefix)
                annotation_entry = {
                    "id": annotation_id_counter,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "value": float_value,
                    "area": 0,
                    "bbox": [],
                    "iscrowd": 0
                }
                coco_output["annotations"].append(annotation_entry)
                annotation_id_counter += 1
            except ValueError:
                # If string cannot be converted to float, skip it
                print(f"Skipping non-numerical string value: {current_key_prefix}: {current_value}")
                pass # Do nothing, effectively skipping this annotation
        elif isinstance(current_value, bool): # Boolean value (treat as 0 or 1)
            cat_id = get_category_id(current_key_prefix)
            annotation_entry = {
                "id": annotation_id_counter,
                "image_id": image_id,
                "category_id": cat_id,
                "value": float(current_value), # Convert bool to float (True=1.0, False=0.0)
                "area": 0,
                "bbox": [],
                "iscrowd": 0
            }
            coco_output["annotations"].append(annotation_entry)
            annotation_id_counter += 1
        elif isinstance(current_value, list):
            for i, item in enumerate(current_value):
                process_and_add_annotations(f"{current_key_prefix}_{i}", item, image_id)
        elif isinstance(current_value, dict):
            for sub_key, sub_value in current_value.items():
                process_and_add_annotations(f"{current_key_prefix}_{sub_key}", sub_value, image_id)

    for data_type in ["Training", "Validation", "test"]:
        search_path = os.path.join(base_dir, data_type, "**", "*.json")
        for json_file in glob.glob(search_path, recursive=True):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Skipping file due to decoding error: {json_file}")
                continue


            image_info = data.get("images", {})
            if not image_info or not data.get("info", {}).get("filename"):
                continue

            image_filename = data.get("info", {}).get("filename")
            # Create a relative path for the image
            relative_image_path = ""
            path_parts = json_file.split(os.sep)
            try:
                dataset_index = path_parts.index("dataset")
                # We want the path relative to the project root, so we include 'dataset'
                relative_image_path = os.path.join(*path_parts[dataset_index:], "..", image_filename)
                # Normalize the path to handle ".."
                relative_image_path = os.path.normpath(relative_image_path)
                # Fix the path to point to origin_data instead of labeled_data
                # Based on user's clarification, it's always 'origin_data'
                relative_image_path = relative_image_path.replace("labeled_data", "origin_data")
            except ValueError:
                # If 'dataset' is not in the path, we can't create a relative path
                # This case should not happen with the current project structure
                relative_image_path = image_filename


            image_entry = {
                "id": image_id_counter,
                "width": image_info.get("width"),
                "height": image_info.get("height"),
                "file_name": relative_image_path,
                "license": None,
                "flickr_url": None,
                "coco_url": None,
                "date_captured": data.get("info", {}).get("date")
            }
            coco_output["images"].append(image_entry)

            annotations = data.get("annotations", {})
            if annotations:
                for key, value in annotations.items():
                    process_and_add_annotations(key, value, image_id_counter)
            
            equipment = data.get("equipment", {})
            if equipment:
                for key, value in equipment.items():
                    process_and_add_annotations(key, value, image_id_counter)

            image_id_counter += 1

    with open(output_file, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print(f"Successfully created COCO dataset at: {output_file}")
    print(f"Total images: {len(coco_output['images'])}")
    print(f"Total annotations: {len(coco_output['annotations'])}")
    print(f"Total categories: {len(coco_output['categories'])}")


if __name__ == "__main__":
    create_coco_dataset()
