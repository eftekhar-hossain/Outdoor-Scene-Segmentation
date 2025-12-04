import torch
import numpy as np


# ================================================================
# Pixel Accuracy
# ================================================================
def pixel_accuracy(logits, targets, device="cpu"):
    """
    Computes pixel-wise accuracy over the entire batch.

    Args:
        logits (Tensor): Raw model output of shape (B, C, H, W)
                         where C = number of classes.
        targets (Tensor): Ground truth mask of shape (B, H, W)
                          with integer class IDs.
        device (str): "cuda" or "cpu"

    Returns:
        float: Pixel accuracy = (# correct pixels / total pixels)
    """

    # Convert logits to predicted class IDs: (B, H, W)
    preds = torch.argmax(logits.to(device), dim=1)

    # Count correct pixel predictions
    correct = (preds == targets).sum().float()

    # Total number of pixels in ground truth mask
    total = torch.numel(targets)

    # Return accuracy as Python float
    return (correct / total).item()



# ================================================================
# Dice / F1 Score (per class -> averaged)
# ================================================================
def dice_f1_score(logits, targets, num_classes, device="cpu"):
    """
    Computes the mean Dice (F1) score across all classes present in the batch.

    The Dice score for class c:
        Dice = 2 * |Prediction ∩ GroundTruth| / (|Pred| + |GT|)
    Args:
        logits (Tensor): Raw model output (B, C, H, W)
        targets (Tensor): Integer mask (B, H, W)
        num_classes (int): Number of segmentation classes
        device (str): Computing device

    Returns:
        float: Mean Dice score across batch
    """

    # Convert logits to predicted class IDs: shape (B, H, W)
    preds = torch.argmax(logits.to(device), dim=1)
    targets = targets.to(device)

    batch_scores = []  # list of mean dice per image

    # ------------------------------------------------------------
    # Process batch element-by-element
    # ------------------------------------------------------------
    for i in range(preds.size(0)):
        pred_mask = preds[i]      # (H, W)
        true_mask = targets[i]    # (H, W)

        class_scores = []  # dice score per valid class

        # --------------------------------------------------------
        # Compute Dice for each class individually
        # --------------------------------------------------------
        for cls in range(num_classes):

            pred_cls = (pred_mask == cls)   # predicted pixels for class cls
            true_cls = (true_mask == cls)   # GT pixels for class cls

            # Skip class if not present in GT
            if true_cls.sum().item() == 0:
                continue

            # Intersection and union-style denominator
            intersection = (pred_cls & true_cls).sum().float()
            denom = pred_cls.sum().float() + true_cls.sum().float()

            # Dice formula
            dice = (2 * intersection) / (denom + 1e-8)
            class_scores.append(dice)

        # If at least 1 class existed, average class dice
        if class_scores:
            class_scores_tensor = torch.stack(class_scores)
            batch_scores.append(class_scores_tensor.mean().item())

    # Average across images in batch
    return float(np.mean(batch_scores))



# ================================================================
# Mean Intersection over Union (mIoU)
# ================================================================
def mean_iou(logits, targets, num_classes, smooth=1e-10):
    """
    Computes mean IoU across all classes.

    IoU for class c:
        IoU = |Prediction ∩ GroundTruth| / |Prediction ∪ GroundTruth|

    Args:
        logits (Tensor): Raw model output (B, C, H, W)
        targets (Tensor): Ground truth mask (B, H, W)
        num_classes (int): Total number of segmentation classes
        smooth (float): Small constant to avoid division-by-zero

    Returns:
        float: Mean IoU across present classes
    """

    # Convert logits -> predicted class IDs
    preds = torch.argmax(logits, dim=1)

    # Flatten batch into 1D vectors for global IoU
    preds = preds.contiguous().view(-1)
    true = targets.contiguous().view(-1)

    iou_scores = []

    # ------------------------------------------------------------
    # Compute IoU per class
    # ------------------------------------------------------------
    for cls in range(num_classes):

        pred_cls = (preds == cls)
        true_cls = (true == cls)

        # Skip class if no ground-truth pixel exists
        if true_cls.sum().item() == 0:
            continue

        # Set intersection and union
        intersection = (pred_cls & true_cls).sum().float()
        union = pred_cls.sum().float() + true_cls.sum().float() - intersection

        # IoU formula
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())

    # Return mean over valid classes
    return float(np.mean(iou_scores)) if iou_scores else 0.0
