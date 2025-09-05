import json
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import multiprocessing



PROFILE_FEATURES = [
    'forehead_moisture', 'l_cheek_moisture', 'r_cheek_moisture', 'chin_moisture',
    'forehead_wrinkle', 'l_perocular_wrinkle', 'r_perocular_wrinkle', 'glabellus_wrinkle',
    'l_cheek_pore', 'r_cheek_pore',
    'l_cheek_pigmentation', 'r_cheek_pigmentation', 'forehead_pigmentation',
    'l_cheek_elasticity_R2', 'r_cheek_elasticity_R2', 'forehead_elasticity_R2', 'chin_elasticity_R2'
]

def calculate_single_person_profile(task):
    """
    한 사람의 데이터를 받아 최종 프로필을 계산하는 작업자(worker) 함수
    """
    person_id, annotations_data, images_list, feature_list = task
    
    profile_vector = []
    for feature in feature_list:
        values = annotations_data.get(feature, [])
        avg_value = np.mean(values) if values else 0.0
        profile_vector.append(avg_value)
        
    return {
        'person_id': person_id,
        'images': images_list,
        'profile': profile_vector
    }

def main():
    json_path = '../dataset_preprocessed/coco_skin_dataset.json'
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    
    print("Pre-organizing data for faster lookup...")
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    image_id_to_person_id = {
        img['id']: os.path.basename(img['file_name']).split('_')[0] 
        for img in coco_data['images']
    }

    
    person_annotations = defaultdict(lambda: defaultdict(list))
    person_images = defaultdict(list)

    print("Grouping annotations and images by person...")
    
    for img in coco_data['images']:
        person_id = image_id_to_person_id.get(img['id'])
        if person_id:
            person_images[person_id].append(img['file_name'])

    
    for ann in tqdm(coco_data['annotations'], desc="Grouping Annotations"):
        person_id = image_id_to_person_id.get(ann['image_id'])
        category_name = cat_id_to_name.get(ann['category_id'])
        
        if person_id and category_name and category_name in PROFILE_FEATURES:
            person_annotations[person_id][category_name].append(float(ann['value']))

    
    tasks = []
    for person_id, annotations_data in person_annotations.items():
        images_list = person_images.get(person_id, [])
        tasks.append((person_id, annotations_data, images_list, PROFILE_FEATURES))
    
    print(f"\nCreated {len(tasks)} tasks for parallel processing.")

    
    num_workers = multiprocessing.cpu_count()
    print(f"Starting profile calculation with {num_workers} CPU cores...")
    
    final_persons_list = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        
        results = tqdm(pool.imap_unordered(calculate_single_person_profile, tasks), total=len(tasks), desc="Calculating Profiles")
        for person_profile in results:
            final_persons_list.append(person_profile)

    
    final_data = {
        'features': PROFILE_FEATURES,
        'persons': sorted(final_persons_list, key=lambda x: x['person_id']) 
    }
    
    save_path = os.path.join(os.path.dirname(json_path), 'person_profiles.json')
    with open(save_path, 'w') as f:
        json.dump(final_data, f, indent=4)
        
    print(f"\nPerson profiles created successfully at: {save_path}")

if __name__ == '__main__':
    main()