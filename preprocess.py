import os
import json
from PIL import Image
from tqdm import tqdm
import shutil
import multiprocessing

def process_image(task):
    original_img_path, new_img_path, new_size = task

    try:
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
        
        with Image.open(original_img_path) as img:
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            resized_img = resized_img.convert('RGB')
            resized_img.save(new_img_path)
        return None 
    except FileNotFoundError:
        return f"Warning: Source image not found at {original_img_path}. Skipping."
    except Exception as e:
        return f"Error processing {original_img_path}: {e}"


def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    new_size=(224, 224)

    json_path = os.path.join(base_path, 'coco_skin_dataset.json')
    output_base_dir = os.path.join(base_path, 'dataset_preprocessed')
    
    if os.path.exists(output_base_dir):
        print(f"'{output_base_dir}' exists. Removing it to start fresh.")
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir)
    print(f"Created directory: {output_base_dir}")

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    tasks = []
    for img_info in coco_data['images']:
        original_img_path = os.path.join(base_path, img_info['file_name'])
        new_img_path = os.path.join(output_base_dir, img_info['file_name'])
        tasks.append((original_img_path, new_img_path, new_size))

    print(f"Preparing to process {len(tasks)} images using multiple cores...")

    num_workers = multiprocessing.cpu_count()
    print(f"Creating a pool of {num_workers} worker processes.")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks), desc="Processing Images"))

    errors = [res for res in results if res is not None]
    if errors:
        print("\n--- Encountered some errors during processing ---")
        for error in errors[:10]:
            print(error)

    new_json_path = os.path.join(output_base_dir, 'coco_skin_dataset.json')
    shutil.copy2(json_path, new_json_path)
    print(f"\nCopied JSON file to {new_json_path}")
    print("Preprocessing finished successfully!")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()