import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset # <<< Subset 추가
import torchvision.models as models
from torchvision import transforms

from tqdm import tqdm

import json
import os
from PIL import Image
import argparse
import numpy as np

# --- 1. '스마트'해진 SkinDataset 클래스 ---
class SkinDataset(Dataset):
    # <<< 'train' 플래그를 받아 훈련용/검증용 변환을 내부에서 결정
    def __init__(self, json_path, train=True):
        print(f"Loading dataset from: {json_path}")
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # --- 훈련용(데이터 증강 포함)과 검증용 변환을 별도로 정의 ---
        self.transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), # 훈련 시에만 데이터 증강 적용
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_val = transforms.Compose([
            transforms.Resize((224, 224)), # 검증 시에는 리사이즈만
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # train 플래그에 따라 사용할 변환을 선택
        self.transform = self.transform_train if train else self.transform_val
        
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

        print(f"Found {len(self.coco_data['images'])} total images.")
        print(f"Using {len(self.valid_images)} images with annotations.")
        print(f"Number of categories to predict: {self.num_categories}")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_info = self.valid_images[idx]
        
        # <<< json 파일 위치를 기준으로 이미지 경로를 동적으로 생성
        base_dir = os.path.dirname(self.json_path)
        img_path = os.path.join(base_dir, img_info['file_name'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path}. Returning a dummy image.")
            image = Image.new('RGB', (224, 224), color = 'red')

        # __init__에서 선택된 transform을 적용
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
    # <<< 2. ssdlite 코드처럼 데이터셋을 훈련용/검증용으로 나누어 생성 ---
    # 훈련용 데이터셋 (데이터 증강 포함)
    dataset_train = SkinDataset(json_path=args.json_path, train=True)
    # 검증용 데이터셋 (데이터 증강 없음)
    dataset_val = SkinDataset(json_path=args.json_path, train=False)
    
    # 인덱스를 섞어서 훈련/검증 데이터 분리
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset_train)).tolist()
    train_size = int(0.8 * len(dataset_train))
    
    # Subset을 사용하여 동일한 인덱스로 데이터셋을 나눔
    subset_train = Subset(dataset_train, indices[:train_size])
    subset_val = Subset(dataset_val, indices[train_size:])

    print(f"Training set size: {len(subset_train)}")
    print(f"Validation set size: {len(subset_val)}")

    # <<< 3. DataLoader를 개선된 방식으로 생성 ---
    # num_workers와 pin_memory 옵션을 추가하여 효율성 증대
    train_loader = DataLoader(subset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(subset_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss, and optimizer
    num_classes = dataset_train.num_categories
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
    parser.add_argument('--json_path', type=str, default='/home/bbang/Workspace/Intel07_Intelproject_Team3/coco_skin_dataset.json', help='Path to the COCO JSON dataset.')
    parser.add_argument('--model_save_path', type=str, default='/home/bbang/Workspace/Intel07_Intelproject_Team3/AI/skin_analysis_model.pth', help='Path to save the trained model.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train.')

    args = parser.parse_args()
    main(args)