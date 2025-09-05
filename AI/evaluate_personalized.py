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
import numpy as np


from model import PersonalizedSkinModel 

class TestDataset(Dataset):
    """테스트를 위한 데이터셋 클래스, 이미지 경로도 함께 반환하도록 수정"""
    def __init__(self, test_dir, person_id_to_idx, transform=None):
        self.transform = transform
        self.samples = []
        
        print(f"Scanning test images in: {test_dir}")
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    person_id = file.split('_')[0]
                    if person_id in person_id_to_idx:
                        img_path = os.path.join(root, file)
                        person_idx = person_id_to_idx[person_id]
                        self.samples.append((img_path, person_idx))
                        
        print(f"Found {len(self.samples)} test images for known persons.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, person_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        
        return image, torch.tensor(person_idx, dtype=torch.long), img_path


def evaluate(args):
    """학습된 모델을 사용하여 테스트셋 전체를 평가하고, 샘플 예측값을 보여줍니다."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading data info from: {args.json_path}")
    with open(args.json_path, 'r') as f:
        profile_data = json.load(f)
    
    feature_list = profile_data['features']
    num_profile_features = len(feature_list)
    
    person_id_to_idx = {person['person_id']: i for i, person in enumerate(profile_data['persons'])}
    num_persons = len(person_id_to_idx)
    
    person_profiles = [None] * num_persons
    for person_data in profile_data['persons']:
        idx = person_id_to_idx[person_data['person_id']]
        person_profiles[idx] = torch.tensor(person_data['profile'], dtype=torch.float32)

    print("Initializing model...")
    
    model = PersonalizedSkinModel(num_persons=num_persons, num_profile_features=num_profile_features)
    
    print(f"Loading trained weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = TestDataset(test_dir=args.test_dir, person_id_to_idx=person_id_to_idx, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_predictions = []
    all_ground_truths = []
    
    print("\nStarting evaluation on the test set...")
    
    
    if args.show_samples > 0:
        print("\n--- Sample Predictions ---")
    samples_shown = 0

    with torch.no_grad():
        for images, person_indices, image_paths in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            person_indices = person_indices.to(device)
            
            predictions = model(images, person_indices)
            ground_truths = torch.stack([person_profiles[idx.item()] for idx in person_indices]).to(device)
            
            
            if samples_shown < args.show_samples:
                num_to_show_this_batch = min(len(images), args.show_samples - samples_shown)
                for i in range(num_to_show_this_batch):
                    print(f"\n[Sample {samples_shown + i + 1}]")
                    print(f"  Image: {os.path.basename(image_paths[i])}")
                    print(f"  Person ID: {list(person_id_to_idx.keys())[person_indices[i].item()]}")
                    print("  --------------------------------------------------")
                    print(f"  {'Feature':<25} | {'Predicted':>10} | {'Actual':>10}")
                    print("  --------------------------------------------------")
                    for j, feature_name in enumerate(feature_list):
                        pred_val = predictions[i, j].item()
                        actual_val = ground_truths[i, j].item()
                        print(f"  {feature_name:<25} | {pred_val:>10.2f} | {actual_val:>10.2f}")
                    print("  --------------------------------------------------")
                samples_shown += num_to_show_this_batch


            all_predictions.append(predictions)
            all_ground_truths.append(ground_truths)

    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_ground_truths_tensor = torch.cat(all_ground_truths, dim=0)
    
    overall_mae = torch.mean(torch.abs(all_predictions_tensor - all_ground_truths_tensor)).item()
    overall_rmse = torch.sqrt(torch.mean((all_predictions_tensor - all_ground_truths_tensor)**2)).item()

    print("\n\n--- Overall Performance ---")
    print(f"- Overall MAE : {overall_mae:.4f}")
    print(f"- Overall RMSE: {overall_rmse:.4f}")
    print("---------------------------\n")

    print("--- Performance by Feature ---")
    for i, feature_name in enumerate(feature_list):
        feature_mae = torch.mean(torch.abs(all_predictions_tensor[:, i] - all_ground_truths_tensor[:, i])).item()
        feature_rmse = torch.sqrt(torch.mean((all_predictions_tensor[:, i] - all_ground_truths_tensor[:, i])**2)).item()
        print(f"- {feature_name:<25} -> MAE: {feature_mae:.4f}, RMSE: {feature_rmse:.4f}")
    print("------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the personalized skin analysis model.')
    
    parser.add_argument('--test-dir', type=str, default='../dataset/Test/origin_data', help='Path to the root directory of test images.')
    parser.add_argument('--model-path', type=str, default='personalized_skin_model.pth', help='Path to the trained model file (.pth).')
    parser.add_argument('--json-path', type=str, default='../dataset_preprocessed/person_profiles.json', help='Path to the person_profiles.json file.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for DataLoader.')
    
    
    parser.add_argument('--show-samples', type=int, default=5, help='Number of sample predictions to display. Default is 5.')
    
    args = parser.parse_args()
    evaluate(args)