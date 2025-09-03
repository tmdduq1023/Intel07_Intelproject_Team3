
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as F

# --- 1. Dataset Class and Transformations ---

class TransformedCocoDetection(CocoDetection):
    """A wrapper for CocoDetection to apply transforms to both image and target."""
    def __init__(self, root, annFile, train=False):
        super(TransformedCocoDetection, self).__init__(root, annFile)
        self.train = train

    def __getitem__(self, idx):
        img, target = super(TransformedCocoDetection, self).__getitem__(idx)
        
        # The target from CocoDetection is a list of dicts, we need to format it
        # Also, apply transformations
        image_id = self.ids[idx]
        
        # Convert PIL image to tensor
        image = F.to_tensor(img)

        # Format the target
        formatted_target = {}
        formatted_target["boxes"] = torch.as_tensor([obj['bbox'] for obj in target], dtype=torch.float32)
        # COCO bbox format is [x, y, width, height], convert to [x1, y1, x2, y2]
        formatted_target["boxes"][:, 2] = formatted_target["boxes"][:, 0] + formatted_target["boxes"][:, 2]
        formatted_target["boxes"][:, 3] = formatted_target["boxes"][:, 1] + formatted_target["boxes"][:, 3]
        
        formatted_target["labels"] = torch.as_tensor([obj['category_id'] for obj in target], dtype=torch.int64)
        formatted_target["image_id"] = torch.tensor([image_id])
        formatted_target["area"] = torch.as_tensor([obj['area'] for obj in target], dtype=torch.float32)
        formatted_target["iscrowd"] = torch.as_tensor([obj['iscrowd'] for obj in target], dtype=torch.int64)

        # Data Augmentation (Random Horizontal Flip)
        if self.train and torch.rand(1) < 0.5:
            image = F.hflip(image)
            boxes = formatted_target["boxes"]
            img_width, _ = F.get_image_size(image)
            boxes[:, [0, 2]] = img_width - boxes[:, [2, 0]]
            formatted_target["boxes"] = boxes

        return image, formatted_target

# --- 2. Model Definition ---

def get_model(num_classes):
    """
    Loads a pre-trained Faster R-CNN model and modifies the classifier head
    for our specific number of classes.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- 3. Utility Functions ---

def collate_fn(batch):
    return tuple(zip(*batch))

# --- 4. Main Training Logic ---

def main():
    # --- Hyperparameters and Setup ---
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {DEVICE}")

    NUM_CLASSES = 13 # 12 categories + 1 background
    BATCH_SIZE = 4
    NUM_EPOCHS = 3
    LEARNING_RATE = 0.005
    
    PROJECT_ROOT = os.getcwd()
    DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset')
    IMAGE_DIR = os.path.join(DATASET_ROOT, 'Training', 'origin_data')
    ANNOTATION_FILE = os.path.join(DATASET_ROOT, 'coco_annotations.json')

    # --- Dataset and DataLoader ---
    dataset = TransformedCocoDetection(root=IMAGE_DIR, annFile=ANNOTATION_FILE, train=True)
    dataset_val = TransformedCocoDetection(root=IMAGE_DIR, annFile=ANNOTATION_FILE, train=False)

    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(0.8 * len(dataset))
    dataset_train = Subset(dataset, indices[:train_size])
    dataset_val = Subset(dataset_val, indices[train_size:])

    print(f"Training samples: {len(dataset_train)}, Validation samples: {len(dataset_val)}")

    data_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # --- Model, Optimizer, etc. ---
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(data_loader_train, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
        for images, targets in pbar:
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            pbar.set_postfix(loss=losses.item())

        lr_scheduler.step()
        print(f"Epoch {epoch+1} Training Loss: {epoch_loss / len(data_loader_train)}")

        # --- Validation Loop (simplified) ---
        model.eval()
        with torch.no_grad():
            pbar_val = tqdm(data_loader_val, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
            for images, _ in pbar_val:
                images = list(image.to(DEVICE) for image in images)
                _ = model(images)

    print("--- Training Finished ---")
    
    model_save_path = os.path.join(PROJECT_ROOT, 'trained_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
