import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from torchvision import transforms

from tqdm import tqdm
import json
import os
from PIL import Image
import argparse
import numpy as np


FEATURES_TO_TRAIN = [
    'lip_dryness',
    'l_perocular_wrinkle',
    'r_perocular_wrinkle',
    'forehead_pigmentation',
    'forehead_moisture'
]

class SingleFeatureDataset(Dataset):
    def __init__(self, json_path, target_feature_name, transform=None):
        self.transform = transform
        self.samples = []
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        cat_name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}
        if target_feature_name not in cat_name_to_id:
            raise ValueError(f"Feature '{target_feature_name}' not found in dataset categories.")
        target_cat_id = cat_name_to_id[target_feature_name]
        image_id_to_path = {img['id']: img['file_name'] for img in coco_data['images']}
        for ann in coco_data['annotations']:
            
            if ann['category_id'] == target_cat_id and 'bbox' in ann and ann['bbox']:
                image_id = ann['image_id']
                img_path = image_id_to_path.get(image_id)
                if img_path:
                    full_path = os.path.join(os.path.dirname(json_path), img_path)
                    self.samples.append((full_path, float(ann['value']), ann['bbox']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target_value, bbox = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        x, y, w, h = bbox
        
        if w > 0 and h > 0:
            cropped_image = image.crop((x, y, x + w, y + h))
        else:
            
            
            cropped_image = image 
        
        if self.transform:
            image_tensor = self.transform(cropped_image)
        else:
            image_tensor = transforms.ToTensor()(cropped_image)
            
        return image_tensor, torch.tensor(target_value, dtype=torch.float32).unsqueeze(0)

class SingleOutputModel(nn.Module):
    def __init__(self):
        super(SingleOutputModel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1))
    def forward(self, x):
        return self.resnet(x)

def train_model_for_feature(target_feature, args):
    """지정된 단일 지표에 대해 모델을 학습시키는 함수"""
    
    print("\n" + "="*80)
    print(f"--- Starting Training for Model: [{target_feature}] ---")
    print("="*80 + "\n")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        full_dataset = SingleFeatureDataset(json_path=args.json_path, target_feature_name=target_feature, transform=transform)
        if len(full_dataset) < 10:
            print(f"Not enough data ({len(full_dataset)} samples) for '{target_feature}'. Skipping.")
            return
    except ValueError as e:
        print(e)
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SingleOutputModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    model_save_path = f"model_{target_feature}.pth"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for i, (inputs, labels) in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss / (i + 1))
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Validation loss improved to {best_val_loss:.4f}. Saving model to {model_save_path}")
        else:
            patience_counter += 1
            print(f"  -> Validation loss did not improve. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break
            
    print(f"\n--- Finished training for model: [{target_feature}] ---")
    print(f"Best model saved at: {model_save_path} with Validation Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train separate models for each skin feature sequentially with early stopping.')
    parser.add_argument('--json_path', type=str, default='../coco_skin_dataset.json')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100, help="Maximum number of epochs.")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--patience', type=int, default=10, help="Number of epochs to wait for improvement before stopping.")

    args = parser.parse_args()

    for feature in FEATURES_TO_TRAIN:
        train_model_for_feature(feature, args)
    
    print("\n" + "="*80)
    print("All training tasks are complete.")
    print("="*80)