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






TARGET_CATEGORIES = [
    'forehead_moisture',     
    'l_cheek_pore',          
    'l_cheek_elasticity_R2', 
    'forehead_wrinkle'       
]




class SkinDataset(Dataset):
    def __init__(self, json_path, target_categories, transform=None):
        print(f"Loading dataset from: {json_path}")
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.transform = transform
        
        
        original_categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        
        self.target_categories_map = {} 
        self.original_cat_id_to_new_idx = {} 
        
        new_idx = 0
        for cat_id, cat_name in original_categories.items():
            if cat_name in target_categories:
                self.target_categories_map[new_idx] = cat_name
                self.original_cat_id_to_new_idx[cat_id] = new_idx
                new_idx += 1
        
        self.num_categories = len(self.target_categories_map)
        print(f"Model will be trained on {self.num_categories} target categories: {list(self.target_categories_map.values())}")
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

        
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

        
        self.valid_images = [img for img in self.images if img['id'] in self.image_id_to_annotations]

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_info = self.valid_images[idx]
        dataset_dir = os.path.dirname(self.json_path)
        img_path = os.path.join(dataset_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        
        target = torch.zeros(self.num_categories, dtype=torch.float32)
        annotations = self.image_id_to_annotations.get(img_info['id'], [])
        
        for ann in annotations:
            cat_id = ann['category_id']
            
            if cat_id in self.original_cat_id_to_new_idx:
                new_idx = self.original_cat_id_to_new_idx[cat_id]
                target[new_idx] = float(ann['value'])

        return image, target




class SkinAnalysisModel(nn.Module):
    def __init__(self, num_outputs):
        super(SkinAnalysisModel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_outputs)
        )
    def forward(self, x):
        return self.resnet(x)

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets):
        elementwise_loss = self.criterion(outputs, targets)
        mask = (targets != 0).float()
        masked_loss = elementwise_loss * mask
        num_valid_elements = mask.sum()
        mean_loss = masked_loss.sum() / (num_valid_elements + 1e-8)
        return mean_loss




def main(args):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    full_dataset = SkinDataset(json_path=args.json_path, target_categories=TARGET_CATEGORIES, transform=transform)
    
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    
    num_classes = full_dataset.num_categories
    model = SkinAnalysisModel(num_outputs=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    criterion = MaskedMSELoss()
    val_criterion = nn.MSELoss(reduction='none') 

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item()) 
            
        
        model.eval()
        per_category_loss = {name: 0.0 for name in TARGET_CATEGORIES}
        per_category_count = {name: 0 for name in TARGET_CATEGORIES}
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                elementwise_loss = val_criterion(outputs, labels)
                mask = (labels != 0).float()
                
                for i, name in enumerate(TARGET_CATEGORIES):
                    category_mask = mask[:, i]
                    per_category_loss[name] += (elementwise_loss[:, i] * category_mask).sum().item()
                    per_category_count[name] += category_mask.sum().item()

        print(f'\nEpoch [{epoch+1}/{args.epochs}] Validation Results:')
        total_loss = 0
        total_count = 0
        for name in TARGET_CATEGORIES:
            count = per_category_count[name]
            if count > 0:
                avg_loss = per_category_loss[name] / count
                print(f'  - {name} Loss: {avg_loss:.4f} (from {int(count)} samples)')
            else:
                print(f'  - {name} Loss: N/A (no valid samples in this validation set)')
            
            total_loss += per_category_loss[name]
            total_count += count
        
        overall_avg_loss = total_loss / total_count if total_count > 0 else 0
        print(f'  - Overall Avg Validation Loss: {overall_avg_loss:.4f}\n')

    torch.save(model.state_dict(), args.model_save_path)
    print(f"Training finished. Model saved to {args.model_save_path}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a skin analysis model on selected features.')
    parser.add_argument('--json_path', type=str, default='../dataset_preprocessed/coco_skin_dataset.json', help='Path to the preprocessed COCO JSON dataset.')
    parser.add_argument('--model_save_path', type=str, default='skin_analysis_final_model.pth', help='Path to save the trained model.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader.')

    args = parser.parse_args()
    main(args)