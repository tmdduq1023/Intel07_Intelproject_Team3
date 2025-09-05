import json
import os
import random
from PIL import Image, ImageDraw


JSON_FILE_PATH = "/home/bbang/Workspace/Intel07_Intelproject_Team3/coco_skin_dataset.json"
BASE_IMAGE_DIR = "/home/bbang/Workspace/Intel07_Intelproject_Team3/dataset"
OUTPUT_DIR = "test_results"  
NUM_SAMPLES = 5  



def test_bbox():
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"'{OUTPUT_DIR}' 폴더를 생성했습니다.")

    
    print(f"'{JSON_FILE_PATH}' 파일을 로드 중입니다...")
    with open(JSON_FILE_PATH, 'r') as f:
        coco_data = json.load(f)
    print("파일 로드 완료.")

    images = coco_data['images']
    annotations = coco_data['annotations']
    
    
    print("\n--- BBox 통계 검사 시작 ---")
    total_anns = len(annotations)
    invalid_format_count = 0
    for ann in annotations:
        bbox = ann.get('bbox')
        
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(val, (int, float)) for val in bbox):
            invalid_format_count += 1
    
    print(f"총 어노테이션 개수: {total_anns}")
    print(f"BBox 형식이 잘못된 어노테이션 개수: {invalid_format_count}")
    if invalid_format_count == 0:
        print("-> 모든 BBox 형식이 유효합니다.")
    print("--- 통계 검사 완료 ---\n")


    
    print(f"--- 시각적 검사 시작 (무작위 샘플 {NUM_SAMPLES}개) ---")
    
    image_map = {img['id']: img for img in images}
    annotation_map = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in annotation_map:
            annotation_map[img_id] = []
        annotation_map[img_id].append(ann)

    
    annotated_image_ids = list(annotation_map.keys())
    if len(annotated_image_ids) < NUM_SAMPLES:
        print(f"[경고] 어노테이션이 있는 이미지가 {len(annotated_image_ids)}개 뿐이라서, 이 개수만큼만 테스트합니다.")
        sample_ids = annotated_image_ids
    else:
        sample_ids = random.sample(annotated_image_ids, NUM_SAMPLES)

    for i, image_id in enumerate(sample_ids):
        image_info = image_map[image_id]
        image_filename = image_info['file_name']
        
        
        image_path = None
        for root, dirs, files in os.walk(BASE_IMAGE_DIR):
            if image_filename in files:
                image_path = os.path.join(root, image_filename)
                break
        
        if not image_path:
            print(f"[{i+1}/{NUM_SAMPLES}] 이미지 파일을 찾을 수 없습니다: {image_filename}")
            continue

        print(f"[{i+1}/{NUM_SAMPLES}] 처리 중: {image_filename}")
        
        
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        
        for ann in annotation_map[image_id]:
            bbox = ann.get('bbox')
            if bbox and len(bbox) == 4:
                x, y, w, h = bbox
                
                draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        
        
        output_path = os.path.join(OUTPUT_DIR, f"test_{image_filename}")
        image.save(output_path)
        print(f" -> 결과 저장: {output_path}")

    print("--- 시각적 검사 완료 ---")


if __name__ == '__main__':
    test_bbox()