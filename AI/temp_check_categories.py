import json
from collections import defaultdict

json_file_path = '/home/bbang/Workspace/Intel07_Intelproject_Team3/dataset/roi_coco_training.json'

try:
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    categories = data.get('categories', [])
    annotations = data.get('annotations', [])

    # Create a mapping from category_id to category_name
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

    # Count annotations per category
    category_counts = defaultdict(int)
    for ann in annotations:
        category_id = ann.get('category_id')
        if category_id in cat_id_to_name:
            category_name = cat_id_to_name[category_id]
            category_counts[category_name] += 1

    print('Annotation counts per category in roi_coco_training.json:')
    if category_counts:
        for cat_name, count in sorted(category_counts.items()):
            print(f'- {cat_name}: {count}')
    else:
        print('No annotations found or categories list is empty.')

except FileNotFoundError:
    print(f'Error: File not found at {json_file_path}')
except json.JSONDecodeError:
    print(f'Error: Could not decode JSON from {json_file_path}. Invalid JSON format.')
except Exception as e:
    print(f'An unexpected error occurred: {e}')