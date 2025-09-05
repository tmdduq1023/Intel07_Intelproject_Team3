
import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_ssdlite_model, get_fasterrcnn_model, TransformedCocoDetection, collate_fn

def main():
    
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {DEVICE}")
    # We have 9 ROI categories + 1 background class
    NUM_CLASSES = 10
    BATCH_SIZE = 8 # Increased for better GPU utilization
    NUM_EPOCHS = 30 # Increased epochs for better convergence
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 0.0005
    
    # Adjusted paths for the new ROI dataset
    PROJECT_ROOT = Path(__file__).resolve().parent.parent # Should be Intel07_Intelproject_Team3
    DATASET_ROOT = PROJECT_ROOT / 'dataset'
    
    # The root image directory is the dataset folder itself, as file paths in COCO json are relative to it.
    IMAGE_DIR = DATASET_ROOT 
    TRAIN_ANNOTATION_FILE = DATASET_ROOT / 'roi_coco_training.json'
    VAL_ANNOTATION_FILE = DATASET_ROOT / 'roi_coco_validation.json'

    # Use the custom dataset class for COCO
    dataset_train = TransformedCocoDetection(root=str(IMAGE_DIR), annFile=str(TRAIN_ANNOTATION_FILE), train=True)
    dataset_val = TransformedCocoDetection(root=str(IMAGE_DIR), annFile=str(VAL_ANNOTATION_FILE), train=False)

    print(f"Training samples: {len(dataset_train)}, Validation samples: {len(dataset_val)}")

    # Reduced num_workers to prevent system freeze from resource exhaustion.
    data_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=20, collate_fn=collate_fn, pin_memory=True)
    data_loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=20, collate_fn=collate_fn, pin_memory=True)

    # Get the Faster R-CNN model
    model = get_fasterrcnn_model(NUM_CLASSES)
    model.to(DEVICE)

    # Optimizer and Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Early Stopping parameters
    PATIENCE = 5
    best_val_loss = float('inf')
    patience_counter = 0

    print("--- Starting Training ---")
    try:
        for epoch in range(NUM_EPOCHS):
            # --- Training Phase ---
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
                pbar.set_postfix(loss=f"{losses.item():.4f}")

            lr_scheduler.step()
            avg_train_loss = epoch_loss / len(data_loader_train)
            print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

            # --- Validation Phase ---
            model.eval()
            val_epoch_loss = 0
            with torch.no_grad():
                pbar_val = tqdm(data_loader_val, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
                for images, targets in pbar_val:
                    images = list(image.to(DEVICE) for image in images)
                    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                    # Temporarily set model to training mode to get losses, but within torch.no_grad()
                    # This is a common way to get validation loss for detection models in torchvision
                    model.train()
                    loss_dict_val = model(images, targets)
                    model.eval() # Switch back to eval mode

                    losses_val = sum(loss for loss in loss_dict_val.values())
                    val_epoch_loss += losses_val.item()
                    pbar_val.set_postfix(loss=f"{losses_val.item():.4f}")
            
            avg_val_loss = val_epoch_loss / len(data_loader_val)
            print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

            # --- Early Stopping Logic ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save the best model
                model_save_path = os.path.join(os.getcwd(), 'best_roi_detector_res50.pth')
                torch.save(model.state_dict(), model_save_path)
                print(f"Validation loss improved. Saving best model to {model_save_path}")
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
                break # Exit the training loop
        
        print("--- Training Finished ---")

    except KeyboardInterrupt:
        print("\n--- Training interrupted by user. Exiting gracefully. ---")
        # A clean exit is needed to ensure data loader worker processes are terminated.
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == "__main__":
    main()
