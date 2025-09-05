import json
import random

JSON_FILE_PATH = "/home/bbang/Workspace/Intel07_Intelproject_Team3/coco_skin_dataset.json"

def diagnose():
    with open(JSON_FILE_PATH, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    
    
    
    target_ann = None
    for ann in annotations:
        if ann.get('bbox'): 
            target_ann = ann
            break
            
    if target_ann:
        print("--- 샘플 어노테이션 데이터 ---")
        print(target_ann)
        print("\n--- BBox 값 및 타입 분석 ---")
        bbox = target_ann.get('bbox')
        if bbox:
            print(f"BBox 값: {bbox}")
            print(f"BBox의 첫 번째 값 타입: {type(bbox[0])}")
    else:
        print("BBox 데이터가 포함된 어노테이션을 찾을 수 없습니다.")

if __name__ == '__main__':
    diagnose()