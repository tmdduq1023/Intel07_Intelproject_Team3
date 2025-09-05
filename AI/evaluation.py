
import torch
import argparse
import os
import csv
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


from utils import get_fasterrcnn_model, get_ssdlite_model, TransformedCocoDetection, collate_fn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def format_predictions_for_coco(image_ids, predictions):
    """Formats model predictions into the list of dicts that COCOeval expects."""
    coco_results = []
    for i, prediction in enumerate(predictions):
        
        if prediction['boxes'].nelement() == 0:
            continue
        image_id = image_ids[i].item() if torch.is_tensor(image_ids[i]) else image_ids[i]

        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            box[2] -= box[0]
            box[3] -= box[1]
            coco_results.append({
                "image_id": image_id,
                "category_id": label.item(),
                "bbox": box.tolist(),
                "score": score.item(),
            })
    return coco_results

def save_results_to_csv(filepath, model_type, model_path, stats):
    """Saves the evaluation metrics to a CSV file."""
    file_exists = os.path.isfile(filepath)
    
    header = [
        'timestamp', 'model_type', 'model_path', 
        'AP_IoU_50_95', 'AP_IoU_50', 'AP_IoU_75', 
        'AP_small', 'AP_medium', 'AP_large', 
        'AR_max_1', 'AR_max_10', 'AR_max_100', 
        'AR_small', 'AR_medium', 'AR_large'
    ]
    
    row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': model_type,
        'model_path': model_path,
        'AP_IoU_50_95': stats[0],
        'AP_IoU_50': stats[1],
        'AP_IoU_75': stats[2],
        'AP_small': stats[3],
        'AP_medium': stats[4],
        'AP_large': stats[5],
        'AR_max_1': stats[6],
        'AR_max_10': stats[7],
        'AR_max_100': stats[8],
        'AR_small': stats[9],
        'AR_medium': stats[10],
        'AR_large': stats[11],
    }

    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results saved to {filepath}")

def main(args):
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {DEVICE}")

    
    NUM_CLASSES = 13
    if args.model_type == 'faster_rcnn':
        model = get_fasterrcnn_model(NUM_CLASSES)
    elif args.model_type == 'ssdlite_mobilenet':
        model = get_ssdlite_model(NUM_CLASSES)
    else:
        raise ValueError("Invalid model type specified.")

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"Model loaded successfully from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return

    
    print("--- WARNING: For a true performance measure, use a separate, unseen test set. ---")
    dataset = TransformedCocoDetection(root=args.data_path, annFile=args.ann_file, train=False)
    
    
    if args.test_all:
        dataset_test = dataset
    else:
        torch.manual_seed(42)
        indices = torch.randperm(len(dataset)).tolist()
        test_size = int(0.1 * len(dataset))
        dataset_test = Subset(dataset, indices[:test_size])
        print(f"--- Evaluating on a random 10% subset ({len(dataset_test)} images). Use --test-all to evaluate on the full dataset. ---")

    data_loader_test = DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print(f"Evaluation dataset size: {len(dataset_test)}")

    
    coco_gt = dataset.coco
    results = []
    
    pbar = tqdm(data_loader_test, desc="Evaluating")
    for images, targets in pbar:
        images = list(img.to(DEVICE) for img in images)
        image_ids = [t['image_id'] for t in targets]

        with torch.no_grad():
            predictions = model(images)
        
        predictions = [{k: v.to('cpu') for k, v in p.items()} for p in predictions]
        results.extend(format_predictions_for_coco(image_ids, predictions))

    if not results:
        print("No predictions were made. Cannot evaluate.")
        return

    
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    
    save_results_to_csv(args.output_csv, args.model_type, args.model_path, coco_eval.stats)

    print("--- Evaluation Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an object detection model.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained .pth model file.')
    parser.add_argument('--model-type', type=str, required=True, choices=['faster_rcnn', 'ssdlite_mobilenet'], help='Type of the model architecture.')
    
    project_root = os.getcwd()
    default_data_path = os.path.join(project_root, 'dataset', 'Training', 'origin_data')
    default_ann_file = os.path.join(project_root, 'dataset', 'coco_annotations.json')
    default_output_csv = os.path.join(project_root, 'evaluation_results.csv')

    parser.add_argument('--data-path', type=str, default=default_data_path, help='Path to the root image directory.')
    parser.add_argument('--ann-file', type=str, default=default_ann_file, help='Path to the COCO annotation file.')
    parser.add_argument('--output-csv', type=str, default=default_output_csv, help='Path to save the evaluation results CSV.')
    parser.add_argument('--test-all', action='store_true', help='Flag to evaluate on the entire dataset instead of a 10% subset.')

    args = parser.parse_args()
    main(args)
