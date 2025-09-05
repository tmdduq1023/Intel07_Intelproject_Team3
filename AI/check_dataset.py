import json
from collections import defaultdict

JSON_PATH = '../coco_skin_dataset.json'

FEATURES_TO_TRAIN = [
    'lip_dryness',
    'l_perocular_wrinkle',
    'r_perocular_wrinkle',
    'forehead_pigmentation',
    'forehead_moisture'
]

def final_diagnostic(json_path, features):
    """
    지정된 feature들에 대해, 학습 코드의 필터링 로직을 단계별로 추적하여
    각 단계에서 몇 개의 샘플이 통과하는지 검사합니다.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로: {json_path}")
        return

    cat_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}
    cat_id_to_name = {v: k for k, v in cat_name_to_id.items()}

    target_cat_ids = {cat_name_to_id.get(name) for name in features if name in cat_name_to_id}

    stats = {name: defaultdict(int) for name in features}

    for ann in data['annotations']:
        cat_id = ann.get('category_id')
        if cat_id not in target_cat_ids:
            continue

        feature_name = cat_id_to_name[cat_id]
        
        stats[feature_name]['step1_total_annotations'] += 1

        if 'bbox' in ann:
            stats[feature_name]['step2_has_bbox_key'] += 1
        else:
            continue 

        if ann['bbox']:
            stats[feature_name]['step3_has_non_empty_bbox'] += 1
        else:
            continue

    print("="*60)
    print("최종 진단: 필터링 단계별 샘플 통과 개수 추적")
    print("="*60)
    for feature in features:
        print(f"\n--- 분석 대상: [ {feature} ] ---")
        s = stats[feature]
        total = s['step1_total_annotations']
        has_key = s['step2_has_bbox_key']
        non_empty = s['step3_has_non_empty_bbox']
        
        print(f" 1. 총 어노테이션 수: {total} 개")
        print(f" 2. 'bbox' 키를 가진 어노테이션 수: {has_key} 개")
        print(f" 3. 'bbox' 값이 비어있지 않은 어노테이션 수: {non_empty} 개")
        
        if total > 0 and non_empty == 0:
            print("\n  [결론] 이 카테고리의 모든 어노테이션이 'bbox' 정보를 가지고 있지 않거나, 값이 비어있습니다.")
            print("         이것이 데이터가 0개로 로드되는 원인입니다.")
        elif total == 0:
            print("\n  [결론] 이 카테고리에 대한 어노테이션을 찾을 수 없습니다.")
        else:
             print(f"\n  [결론] 최종적으로 유효한 샘플 {non_empty} 개를 찾았습니다. 코드가 정상 동작해야 합니다.")


if __name__ == '__main__':
    final_diagnostic(JSON_PATH, FEATURES_TO_TRAIN)