
import os
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


from utils import get_ssdlite_model, TransformedCocoDetection, collate_fn


def main():
    
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {DEVICE}")

    NUM_CLASSES = 13 
    BATCH_SIZE = 16 
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.005
    
    PROJECT_ROOT = os.getcwd()
    DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset')
    IMAGE_DIR = os.path.join(DATASET_ROOT, 'Training', 'origin_data')
    ANNOTATION_FILE = os.path.join(DATASET_ROOT, 'coco_annotations.json')

    
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

    
    model = get_ssdlite_model(NUM_CLASSES)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    
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

        
        model.eval()
        with torch.no_grad():
            pbar_val = tqdm(data_loader_val, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
            for images, _ in pbar_val:
                images = list(image.to(DEVICE) for image in images)
                _ = model(images)

    print("--- Training Finished ---")
    
    model_save_path = os.path.join(PROJECT_ROOT, 'trained_ssdlite_mobilenet.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
