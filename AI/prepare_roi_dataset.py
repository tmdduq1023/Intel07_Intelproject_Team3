
import json
from pathlib import Path
import argparse

def filter_coco_for_roi(input_json_path, output_json_path):
    """
    Filters a COCO annotation file to keep only 'facepart::*' categories and annotations,
    and remaps category IDs to be contiguous starting from 1.
    It also filters images to only keep those that have at least one ROI annotation.
    """
    print(f"Reading {input_json_path}...")
    p = Path(input_json_path)
    if not p.exists():
        print(f"Error: Input file not found at {p}")
        return

    coco_data = json.loads(p.read_text(encoding="utf-8"))

    # 1. Filter categories to keep only those starting with 'facepart::'
    roi_categories = [cat for cat in coco_data['categories'] if cat['name'].startswith('facepart::')]
    
    if not roi_categories:
        print("No 'facepart::' categories found in the file.")
        return

    # 2. Create a mapping from old category_id to new category_id (1, 2, 3, ...)
    old_cat_ids = {cat['id'] for cat in roi_categories}
    cat_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(list(old_cat_ids)), 1)}

    # 3. Update category IDs in the filtered list
    for cat in roi_categories:
        cat['id'] = cat_id_map[cat['id']]

    # 4. Filter annotations and get a set of image IDs that have ROI annotations
    roi_annotations = []
    valid_image_ids = set()
    for ann in coco_data['annotations']:
        if ann.get('category_id') in old_cat_ids:
            ann['category_id'] = cat_id_map[ann['category_id']]
            roi_annotations.append(ann)
            valid_image_ids.add(ann['image_id'])

    # 5. Filter images to keep only those with valid annotations
    roi_images = [img for img in coco_data['images'] if img['id'] in valid_image_ids]

    # 6. Create the new COCO data structure
    filtered_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'images': roi_images,
        'annotations': roi_annotations,
        'categories': roi_categories
    }
    
    filtered_coco['info']['description'] = f"Filtered for ROI detection from {p.name}. Contains only images with ROI annotations."

    # 7. Save the new JSON file
    out_p = Path(output_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(filtered_coco, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"Successfully created filtered dataset at {out_p}")
    print(f"Found {len(roi_images)} images, {len(roi_categories)} ROI categories, and {len(roi_annotations)} annotations.")

def main():
    ap = argparse.ArgumentParser(description="Filter COCO JSON for ROI detection training.")
    ap.add_argument("--input-json", required=True, help="Path to the input merged_*.json file.")
    ap.add_argument("--output-json", required=True, help="Path to save the filtered output coco.json file.")
    args = ap.parse_args()
    
    filter_coco_for_roi(args.input_json, args.output_json)

if __name__ == "__main__":
    main()
