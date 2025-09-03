import os
import json
from PIL import Image
from tqdm import tqdm
import shutil

def preprocess_images(base_path, new_size=(224, 224)):
    """
    COCO JSON 파일을 기반으로 이미지들을 리사이즈하여 새 폴더에 저장함
    JSON 파일도 함께 복사됨
    """

    json_path = os.path.join(base_path, 'coco_skin_dataset.json') 
    
    # 전처리된 데이터를 저장할 새 폴더
    output_base_dir = os.path.join(base_path, 'dataset_preprocessed')
    
    # --- 폴더 생성 ---
    if os.path.exists(output_base_dir):
        print(f"'{output_base_dir}' exists. Removing it to start fresh.")
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir)
    print(f"Created directory: {output_base_dir}")

    # --- JSON 파일 로드 ---
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    print(f"Starting preprocessing of {len(coco_data['images'])} images...")

    # --- 이미지 리사이즈 및 저장 ---
    for img_info in tqdm(coco_data['images'], desc="Resizing Images"):
        original_img_path = os.path.join(base_path, img_info['file_name'])
        new_img_path = os.path.join(output_base_dir, img_info['file_name'])

        # 이미지 파일이 저장될 하위 폴더 생성
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)

        try:
            with Image.open(original_img_path) as img:
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                resized_img = resized_img.convert('RGB')
                resized_img.save(new_img_path)
        except FileNotFoundError:
            print(f"Warning: Source image not found at {original_img_path}. Skipping.")
    
    # --- JSON 파일 복사 ---
    new_json_path = os.path.join(output_base_dir, 'coco_skin_dataset.json')
    shutil.copy2(json_path, new_json_path)
    print(f"Copied JSON file to {new_json_path}")

    print("\nPreprocessing finished successfully!")


if __name__ == '__main__':
    # 이 스크립트가 있는 폴더(프로젝트 루트)를 기준으로 경로를 설정
    project_base_path = os.path.dirname(os.path.abspath(__file__))
    preprocess_images(base_path=project_base_path)