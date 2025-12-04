import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# Import custom modules
from unet import UNET
from att_unet import AttentionUNet
from dataloader import get_dataloaders
from trainer import fit, evaluate_on_test
from utils import plot_training_history


# ===============================================================
#                   CREATE REQUIRED DIRECTORIES
# ===============================================================
# Get the directory where THIS script lives
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go one level up -> project root
root_dir = os.path.dirname(script_dir)

print(f"[INFO] Root directory set to: {root_dir}")
figures_dir = os.path.join(root_dir, "plots")
models_dir = os.path.join(root_dir, "saved_models")

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


# ===============================================================
#                     MAIN FUNCTION
# ===============================================================
def main(args):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {DEVICE}")

    # ------------------------------------------------------------
    # Load DataLoaders
    # ------------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        path=args.data_root,
        batch_size=args.batch_size
    )

    # ------------------------------------------------------------
    # Select Model Type
    # ------------------------------------------------------------
    if args.model_type.lower() == "unet":
        model = UNET(in_channels=3, out_channels=args.num_classes)
        print("\n>>> Using Standard UNet")

    elif args.model_type.lower() == "attunet":
        model = AttentionUNet(in_channels=3, out_channels=args.num_classes)
        print("\n>>> Using Attention UNet")

    else:
        raise ValueError("Invalid model_type. Choose: 'unet' or 'attunet'")

    model = model.to(DEVICE)
    #print(model)

    # ------------------------------------------------------------
    # Loss, Optimizer, Scheduler
    # ------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )

    # ------------------------------------------------------------
    # Train Model
    # ------------------------------------------------------------
    print(f"\n===== TRAINING STARTED ({args.model_type.upper()}) =====")

    history = fit(
        epochs=args.epochs,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_classes=args.num_classes,
        model_name=args.model_type,
        saved_path=models_dir,
        device=DEVICE
    )

    # Save training logs
    history_df = pd.DataFrame({'train_loss': history['train_loss'],
                           'val_loss': history['val_loss'],
                            'train_acc': history['train_acc'],
                           'val_acc': history['val_acc'],
                           'train_miou': history['train_miou'],
                           'val_miou': history['val_miou'],
                           'train_dice': history['train_dice'],
                           'val_dice': history['val_dice']})
    history_df.to_excel(os.path.join(root_dir,f"training_history_{args.model_type}.xlsx"), index=False)

    # ------------------------------------------------------------
    # Plot curves and save
    # ------------------------------------------------------------
    plot_training_history(history, args.model_type, figures_dir)

    # ------------------------------------------------------------
    # TEST SET EVALUATION
    # ------------------------------------------------------------
    print("\n===== TEST SET EVALUATION =====")

    # Load best checkpoint
    best_model_path = os.path.join(models_dir, f"{args.model_type}_model.pt")
    model.load_state_dict(torch.load(best_model_path))
    model.to(DEVICE)

    test_metrics = evaluate_on_test(
        model=model,
        test_loader=test_loader,
        num_classes=args.num_classes,
        device=DEVICE
    )

    print("\n>>> FINAL TEST METRICS <<<")
    print(f"Test Accuracy: {test_metrics['test_acc']:.4f}")
    print(f"Test Dice Coefficient: {test_metrics['test_dice']:.4f}")
    print(f"Test mIoU: {test_metrics['test_miou']:.4f}")
   


# ===============================================================
#                     COMMAND LINE ARGS
# ===============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train UNet or Attention UNet")

    parser.add_argument("--data_root", type=str, default="data-urban",
                        help="Dataset directory with train/val/test folders")
    
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=12,help="Number of segmentation classes")

    parser.add_argument("--model_type",type=str,default="unet", choices=["unet", "attunet"],
        help="Choose segmentation model: 'unet' or 'attunet'"
    )

    args = parser.parse_args()
    main(args)
