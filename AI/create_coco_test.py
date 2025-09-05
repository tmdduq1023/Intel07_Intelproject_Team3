import os
import json
from tqdm import tqdm

def create_coco_from_jsons(root_dir, output_path):
    """
    지정된 디렉토리 하위의 모든 개별 JSON 파일을 읽어
    하나의 COCO 포맷 JSON 파일로 병합합니다.
    """
    
    
    coco_output = {
        "info": {
            "description": "Merged Test Dataset",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    
    image_id_counter = 1
    annotation_id_counter = 1
    category_id_counter = 1
    
    processed_images = {}  
    category_map = {}      

    
    json_files_to_process = []
    print(f"Scanning for JSON files in '{root_dir}'...")
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.json'):
                json_files_to_process.append(os.path.join(subdir, file))
    
    print(f"Found {len(json_files_to_process)} JSON files to merge.")

    
    for json_path in tqdm(json_files_to_process, desc="Merging JSON files"):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        
        image_filename_from_info = data.get("info", {}).get("filename")
        if not image_filename_from_info:
            continue 

        
        if image_filename_from_info not in processed_images:
            current_image_id = image_id_counter
            processed_images[image_filename_from_info] = current_image_id
            
            image_info = data.get("images", {})
            
            
            relative_path = os.path.join(
                *json_path.split(os.sep)[-4:-1], 
                image_filename_from_info
            )
            
            coco_output["images"].append({
                "id": current_image_id,
                "width": image_info.get("width"),
                "height": image_info.get("height"),
                "file_name": relative_path.replace("\\", "/"), 
            })
            image_id_counter += 1
        else:
            current_image_id = processed_images[image_filename_from_info]
        
        
        
        annotations = data.get("annotations", {}) or {}
        equipment = data.get("equipment", {}) or {}
        merged_annotations = {**annotations, **equipment}

        for category_name, value in merged_annotations.items():
            if value is None:
                continue 

            