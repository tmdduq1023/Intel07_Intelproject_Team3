import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import argparse
import os


class SingleOutputModel(nn.Module):
    def __init__(self):
        super(SingleOutputModel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.resnet(x)



MODELS_TO_LOAD = {
    'lip_dryness': 'model_lip_dryness.pth',
    'l_perocular_wrinkle': 'model_l_perocular_wrinkle.pth',
    'r_perocular_wrinkle': 'model_r_perocular_wrinkle.pth',
    'forehead_pigmentation': 'model_forehead_pigmentation.pth',
    'forehead_moisture': 'model_forehead_moisture.pth'
}

def predict(image_path):
    """
    단일 이미지에 대해 5개의 모델을 모두 사용하여 각 지표를 추론합니다.
    """
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return

    
    predictions = {}

    print("Loading models and performing inference...")

    
    for feature_name, model_path in MODELS_TO_LOAD.items():
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found for '{feature_name}' at '{model_path}'. Skipping.")
            predictions[feature_name] = "N/A (Model file not found)"
            continue

        
        model = SingleOutputModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() 

        with torch.no_grad():
            output = model(image_tensor)
            
            predicted_value = output.item()
            predictions[feature_name] = predicted_value

    
    print("\n" + "="*40)
    print("      Skin Analysis Result")
    print("="*40)
    print(f"Image File: {os.path.basename(image_path)}")
    print("-"*40)
    
    for feature_name, value in predictions.items():
        if isinstance(value, float):
            print(f"- {feature_name:<25}: {value:.2f}")
        else:
            print(f"- {feature_name:<25}: {value}")
            
    print("-"*40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict 5 skin features from a single image.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image for prediction.')
    
    args = parser.parse_args()
    predict(args.image)