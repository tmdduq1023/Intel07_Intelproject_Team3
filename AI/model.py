import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models
import json
import os
from PIL import Image


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