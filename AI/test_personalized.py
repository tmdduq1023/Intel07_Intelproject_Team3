import torch
from torchvision import transforms
from PIL import Image
import argparse
import json
import os


from model import PersonalizedSkinModel, PersonDataset

def predict(args):
    """학습된 모델을 사용하여 단일 이미지에 대한 피부 프로필을 예측합니다."""
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    print(f"Loading data info from: {args.json_path}")
    with open(args.json_path, 'r') as f:
        profile_data = json.load(f)
    
    feature_list = profile_data['features']
    num_profile_features = len(feature_list)
    
    
    person_id_to_idx = {person['person_id']: i for i, person in enumerate(profile_data['persons'])}
    num_persons = len(person_id_to_idx)

    
    if args.person_id not in person_id_to_idx:
        print(f"Error: Person ID '{args.person_id}' not found in the training data.")
        return

    
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
    
    print(f"Loading and preprocessing image: {args.image_path}")
    image = Image.open(args.image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device) 

    
    person_idx = person_id_to_idx[args.person_id]
    person_id_tensor = torch.tensor([person_idx], dtype=torch.long).to(device)

    
    print("\nPerforming inference...")
    with torch.no_grad(): 
        predictions = model(image_tensor, person_id_tensor)
        
    
    print("\n--- Predicted Skin Profile ---")
    print(f"Person ID: {args.person_id}")
    print(f"Image: {os.path.basename(args.image_path)}")
    print("------------------------------")
    
    
    predicted_values = predictions.squeeze().cpu().numpy()
    
    for feature_name, value in zip(feature_list, predicted_values):
        print(f"- {feature_name}: {value:.4f}")
    print("------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a personalized skin analysis model.')
    
    parser.add_argument('--image-path', type=str, required=True, help='Path to the input image for testing.')
    parser.add_argument('--person-id', type=str, required=True, help='The ID of the person in the image (e.g., "0002").')
    parser.add_argument('--model-path', type=str, default='personalized_skin_model.pth', help='Path to the trained model file (.pth).')
    parser.add_argument('--json-path', type=str, default='../dataset_preprocessed/person_profiles.json', help='Path to the person_profiles.json file.')
    
    args = parser.parse_args()
    predict(args)