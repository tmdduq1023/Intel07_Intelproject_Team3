import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from torchvision import transforms

import json
import os
from PIL import Image
import argparse
from tqdm import tqdm


class PersonDataset(Dataset):
    def __init__(self, json_path, dataset_dir, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.feature_list = self.data['features']
        
        self.all_samples = []
        self.person_id_to_idx = {person_info['person_id']: i for i, person_info in enumerate(self.data['persons'])}
        
        for person_info in self.data['persons']:
            person_idx = self.person_id_to_idx[person_info['person_id']]
            profile_tensor = torch.tensor(person_info['profile'], dtype=torch.float32)
            for img_filename in person_info['images']:
                img_path = os.path.join(self.dataset_dir, img_filename)
                self.all_samples.append((img_path, person_idx, profile_tensor))

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        img_path, person_idx, profile_tensor = self.all_samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(person_idx, dtype=torch.long), profile_tensor

class PersonalizedSkinModel(nn.Module):
    def __init__(self, num_persons, num_profile_features, embedding_dim=32):
        super(PersonalizedSkinModel, self).__init__()
        self.resnet_backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet_backbone.fc.in_features
        self.resnet_backbone.fc = nn.Identity()

        self.person_embedding = nn.Embedding(num_persons, embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(num_ftrs + embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_profile_features)
        )

    def forward(self, image, person_id):
        image_features = self.resnet_backbone(image)
        person_features = self.person_embedding(person_id)
        combined_features = torch.cat([image_features, person_features], dim=1)
        output = self.mlp(combined_features)
        return output

def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = PersonDataset(
        json_path=args.json_path, 
        dataset_dir=args.image_dir, 
        transform=transform
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    num_persons = len(full_dataset.person_id_to_idx)
    num_profile_features = len(full_dataset.feature_list)
    model = PersonalizedSkinModel(num_persons, num_profile_features)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    print("Starting personalized model training...")
    for epoch in range(args.epochs):
        model.train()
        
        
        running_loss = 0.0
        
        
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for i, (image, person_id, target_profile) in train_bar:
            image, person_id, target_profile = image.to(device), person_id.to(device), target_profile.to(device)
            
            optimizer.zero_grad()
            outputs = model(image, person_id)
            loss = criterion(outputs, target_profile)
            loss.backward()
            optimizer.step()

            
            running_loss += loss.item()
            
            train_bar.set_postfix(loss=running_loss / (i + 1))
            
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for image, person_id, target_profile in val_loader:
                image, person_id, target_profile = image.to(device), person_id.to(device), target_profile.to(device)
                outputs = model(image, person_id)
                loss = criterion(outputs, target_profile)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), args.model_save_path)
    print(f"Training finished. Model saved to {args.model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a personalized skin analysis model.')
    parser.add_argument('--json_path', type=str, default='../dataset_preprocessed/person_profiles.json')
    parser.add_argument('--image_dir', type=str, default='../dataset_preprocessed')
    parser.add_argument('--model_save_path', type=str, default='personalized_skin_model.pth')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader.')

    args = parser.parse_args()
    main(args)