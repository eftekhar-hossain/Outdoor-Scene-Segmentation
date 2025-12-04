import torch
import os
import numpy as np
from tqdm import tqdm
from metrics import pixel_accuracy, dice_f1_score, mean_iou


# ======================================================================
# Get current learning rate from optimizer
# ======================================================================
def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


# ======================================================================
# Main Training Loop
# ======================================================================
def fit( epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, num_classes, model_name, saved_path, device="cuda"):
    """
    Trains a semantic segmentation model using a supervised learning loop.
    The best model (highest validation mIoU) is automatically saved.

    Args:
        epochs (int): Total number of epochs to train.
        model (nn.Module): UNet or other segmentation model.
        train_loader (DataLoader): Batches of training images and masks.
        val_loader (DataLoader): Batches of validation images and masks.
        criterion: Loss function (CrossEntropyLoss recommended).
        optimizer: Optimizer (AdamW recommended).
        scheduler: Learning rate scheduler (OneCycleLR recommended).
        num_classes (int): Total number of segmentation classes.
        model_name (str): Base name for saving best model checkpoint.
        saved_path (str): Directory to save the best model.
        device (str): "cuda" or "cpu".

    Returns:
        dict: Training history containing loss + metric traces.
    """

    # Move model to correct device
    model.to(device)

    # History dictionary for logging
    history = {
        "train_loss": [], "val_loss": [],
        "train_miou": [], "val_miou": [],
        "train_dice": [], "val_dice": [],
        "train_acc":  [], "val_acc":  [],
        "lrs": []   # learning rate trace (for debugging OneCycle)
    }

    # Keep track of best validation mIoU
    best_val_miou = 0.0

    # ==================================================================
    # Epoch loop
    # ==================================================================
    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")

        # ==============================================================
        # TRAINING PHASE
        # ==============================================================
        model.train()
        train_loss = train_miou = train_dice = train_acc = 0

        for images, masks in tqdm(train_loader, desc="Training", ncols=100):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(images)
            loss = criterion(logits, masks)

            # Backward + Step
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update metrics
            train_loss += loss.item()
            train_miou += mean_iou(logits, masks, num_classes)
            train_dice += dice_f1_score(logits, masks, num_classes)
            train_acc  += pixel_accuracy(logits, masks, device)

            # Track LR for debugging purposes
            history["lrs"].append(get_lr(optimizer))

        # Average metrics over the epoch
        train_loss /= len(train_loader)
        train_miou /= len(train_loader)
        train_dice /= len(train_loader)
        train_acc  /= len(train_loader)

        history["train_loss"].append(train_loss)
        history["train_miou"].append(train_miou)
        history["train_dice"].append(train_dice)
        history["train_acc"].append(train_acc)

        print(f"[Train] Loss={train_loss:.4f} | mIoU={train_miou:.4f} | "
              f"Dice={train_dice:.4f} | Acc={train_acc:.4f}")

        # ==============================================================
        # VALIDATION PHASE
        # ==============================================================
        model.eval()
        val_loss = val_miou = val_dice = val_acc = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation", ncols=100):
                images, masks = images.to(device), masks.to(device)

                # Forward pass
                logits = model(images)
                loss = criterion(logits, masks)

                # Update metrics
                val_loss += loss.item()
                val_miou += mean_iou(logits, masks, num_classes)
                val_dice += dice_f1_score(logits, masks, num_classes)
                val_acc  += pixel_accuracy(logits, masks, device)

        # Average validation metrics
        val_loss /= len(val_loader)
        val_miou /= len(val_loader)
        val_dice /= len(val_loader)
        val_acc  /= len(val_loader)

        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        history["val_dice"].append(val_dice)
        history["val_acc"].append(val_acc)

        print(f"[Val] Loss={val_loss:.4f} | mIoU={val_miou:.4f} | "
              f"Dice={val_dice:.4f} | Acc={val_acc:.4f}")

        # --------------------------------------------------------------
        # Save best model based on validation mIoU
        # --------------------------------------------------------------
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            save_path = os.path.join(saved_path,f"{model_name}_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model → {save_path} (mIoU={val_miou:.4f})")

    return history


# ============================================================
# Test Evaluation
# ============================================================
def evaluate_on_test(model, test_loader, num_classes, device):
    """
    Evaluates the trained model on test images.

    Args:
        model: Trained segmentation model (UNet).
        test_loader: DataLoader for testing.
        num_classes: Number of segmentation classes.
        device: Device for evaluation (cpu/cuda).

    Returns:
        dict: { test_acc, test_dice, test_miou }
    """
    model.eval()
    accs, dices, mious = [], [], []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)

            accs.append(pixel_accuracy(logits, masks, device))
            dices.append(dice_f1_score(logits, masks, num_classes, device))
            mious.append(mean_iou(logits, masks, num_classes))

    return {
        "test_acc":  float(np.mean(accs)),
        "test_dice": float(np.mean(dices)),
        "test_miou": float(np.mean(mious)),
    }

