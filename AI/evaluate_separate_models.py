'''
python evaluate_separate_models.py \
    --target-feature 'forehead_moisture' \
    --model-path 'model_forehead_moisture.pth' \
    --show-samples 10
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms

from tqdm import tqdm
import json
import os
from PIL import Image
import argparse


class SingleFeatureDataset(Dataset):
    
    def __init__(self, json_path, target_feature_name, transform=None):
        self.transform = transform
        self.samples = []
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        cat_name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}
        if target_feature_name not in cat_name_to_id:
            raise ValueError(f"Feature '{target_feature_name}' not found.")
        target_cat_id = cat_name_to_id[target_feature_name]
        image_id_to_path = {img['id']: img['file_name'] for img in coco_data['images']}
        for ann in coco_data['annotations']:
            if ann['category_id'] == target_cat_id and float(ann['value']) != 0:
                img_path_from_json = image_id_to_path.get(ann['image_id'])
                if img_path_from_json:
                    
                    
                    full_path = os.path.join(os.path.dirname(json_path), img_path_from_json)
                    self.samples.append((full_path, float(ann['value'])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target_value = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(target_value, dtype=torch.float32).unsqueeze(0), img_path

class SingleOutputModel(nn.Module):
    def __init__(self):
        super(SingleOutputModel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1))
    def forward(self, x):
        return self.resnet(x)

def evaluate(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    test_dataset = SingleFeatureDataset(json_path=args.json_path, target_feature_name=args.target_feature, transform=transform)
    
    if len(test_dataset) == 0:
        print(f"No test data found for '{args.target_feature}'.")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SingleOutputModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    
    print("\nStarting evaluation...")
    
    
    if args.show_samples > 0:
        print("\n--- Sample Predictions ---")
    samples_shown = 0

    with torch.no_grad():
        for inputs, labels, image_paths in tqdm(test_loader, desc=f"Evaluating '{args.target_feature}'"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            
            if samples_shown < args.show_samples:
                num_to_show_this_batch = min(len(inputs), args.show_samples - samples_shown)
                for i in range(num_to_show_this_batch):
                    pred_val = outputs[i].item()
                    actual_val = labels[i].item()
                    image_name = os.path.basename(image_paths[i])
                    print(f"  - Sample {samples_shown + i + 1}: Img: {image_name}, Predicted: {pred_val:.4f}, Actual: {actual_val:.4f}")
                samples_shown += num_to_show_this_batch

            all_preds.append(outputs)
            all_labels.append(labels)

    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)

    mae = torch.mean(torch.abs(all_preds_tensor - all_labels_tensor)).item()
    rmse = torch.sqrt(torch.mean((all_preds_tensor - all_labels_tensor)**2)).item()

    print(f"\n--- Evaluation Result for: {args.target_feature} ---")
    print(f"Model: {args.model_path}")
    print(f"Evaluated on {len(test_dataset)} samples.")
    print(f"- MAE : {mae:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    print("-------------------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a separately trained model.')
    parser.add_argument('--target-feature', type=str, required=True, help='The name of the feature to evaluate.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model file for the feature.')
    parser.add_argument('--json_path', type=str, default='../dataset_preprocessed/coco_skin_dataset.json', help='Path to annotation file.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    
    
    parser.add_argument('--show-samples', type=int, default=5, help='Number of sample predictions to display. Default is 5.')
    
    args = parser.parse_args()
    evaluate(args)