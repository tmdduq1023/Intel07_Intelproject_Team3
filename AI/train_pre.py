import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms

from tqdm import tqdm

import json
import os
from PIL import Image
import argparse
import numpy as np

class SkinDataset(Dataset):
    def __init__(self, json_path, transform=None):
        print(f"Loading dataset from: {json_path}")
        self.json_path = json_path # 나중에 이미지 경로를 만들기 위해 저장
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.transform = transform
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.num_categories = len(self.categories)

        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

        self.valid_images = []
        for img_info in self.images:
            if img_info['id'] in self.image_id_to_annotations:
                 self.valid_images.append(img_info)

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_info = self.valid_images[idx]
        
        # <<< 중요: JSON 파일이 위치한 폴더를 기준으로 이미지 경로 생성
        dataset_dir = os.path.dirname(self.json_path)
        img_path = os.path.join(dataset_dir, img_info['file_name'])
        
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        target = torch.zeros(self.num_categories, dtype=torch.float32)
        annotations = self.image_id_to_annotations.get(img_info['id'], [])
        
        for ann in annotations:
            cat_id = ann['category_id']
            target_idx = cat_id - 1 
            if 0 <= target_idx < self.num_categories:
                target[target_idx] = float(ann['value'])

        return image, target

class SkinAnalysisModel(nn.Module):
    # (이 부분은 변경 없음)
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

def main(args):
    # <<< 중요: transforms.Resize 제거! ---
    transform = transforms.Compose([
        # transforms.Resize((224, 224)), # 더 이상 필요 없으므로 제거
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = SkinDataset(json_path=args.json_path, transform=transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # 데이터 로딩 효율을 위해 num_workers와 pin_memory 사용
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # (이후 학습 로직은 변경 없음)
    num_classes = full_dataset.num_categories
    model = SkinAnalysisModel(num_outputs=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    criterion = nn.MSELoss()
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
            train_bar.set_postfix(loss=running_loss / (train_bar.n + 1))
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_bar.set_postfix(loss=val_loss / (val_bar.n + 1))
        print(f'Epoch [{epoch+1}/{args.epochs}], Validation Loss: {val_loss / len(val_loader):.4f}')
    torch.save(model.state_dict(), args.model_save_path)
    print(f"Training finished. Model saved to {args.model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a skin analysis model.')
    # <<< 중요: 기본 경로를 전처리된 데이터셋으로 변경
    parser.add_argument('--json_path', type=str, default='../dataset_preprocessed/coco_skin_dataset.json', help='Path to the preprocessed COCO JSON dataset.')
    parser.add_argument('--model_save_path', type=str, default='skin_analysis_model.pth', help='Path to save the trained model.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train.')

    args = parser.parse_args()
    main(args)