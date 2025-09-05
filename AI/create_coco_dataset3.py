import json
import os
import glob

def create_coco_dataset():
    base_dir = "/home/bbang/Workspace/Intel07_Intelproject_Team3/dataset"
    output_file = "/home/bbang/Workspace/Intel07_Intelproject_Team3/coco_skin_dataset.json"
    
    coco_output = {
        "info": {
            "description": "Skin Analysis Dataset",
            "version": "2.0",
            "year": 2025,
            "contributor": "Bbang",
            "date_created": "2025/09/04"
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
                "supercategory": "facepart"
            })
            category_id_counter += 1
        return category_map[name]

    image_id_counter = 1
    annotation_id_counter = 1

    
    for data_type in ["Training", "Validation", "Test"]:
        search_path = os.path.join(base_dir, data_type, "labeled_data", "**", "*.json")
        for json_file in glob.glob(search_path, recursive=True):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                print(f"Skipping file due to decoding error: {json_file}")
                continue

            image_info = data.get("images", {})
            info_data = data.get("info", {})
            if not image_info or not info_data.get("filename"):
                continue

            image_filename = info_data["filename"]
            width = image_info.get("width")
            height = image_info.get("height")

            
            image_entry = {
                "id": image_id_counter,
                "width": width,
                "height": height,
                "file_name": image_filename,  
                "license": None,
                "flickr_url": None,
                "coco_url": None,
                "date_captured": info_data.get("date")
            }
            coco_output["images"].append(image_entry)

            
            bbox = image_info.get("bbox")
            facepart_id = image_info.get("facepart")
            category_name = f"facepart_{facepart_id}"
            cat_id = get_category_id(category_name)

            
            attributes = {}
            if "annotations" in data and data["annotations"]:
                attributes.update(data["annotations"])
            if "equipment" in data and data["equipment"]:
                attributes.update(data["equipment"])

            annotation_entry = {
                "id": annotation_id_counter,
                "image_id": image_id_counter,
                "category_id": cat_id,
                "bbox": bbox if bbox else [0, 0, 0, 0],
                "area": bbox[2] * bbox[3] if bbox else 0,
                "iscrowd": 0,
                "attributes": attributes  
            }
            coco_output["annotations"].append(annotation_entry)
            annotation_id_counter += 1

            image_id_counter += 1

    
    with open(output_file, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print(f"✅ Successfully created COCO dataset at: {output_file}")
    print(f"Total images: {len(coco_output['images'])}")
    print(f"Total annotations: {len(coco_output['annotations'])}")
    print(f"Total categories: {len(coco_output['categories'])}")


if __name__ == "__main__":
    create_coco_dataset()
