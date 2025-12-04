import argparse
import os
import torch
import pandas as pd
import segmentation_models_pytorch as smp

# Custom imports
from dataloader import get_dataloaders
from trainer import fit, evaluate_on_test


# ============================================================================
#  DIRECTORY SETUP
# ============================================================================
# Get the directory where THIS script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up → project root
root_dir = os.path.dirname(script_dir)

print(f"[INFO] Root directory set to: {root_dir}")
figures_dir = os.path.join(root_dir, "plots")
models_dir = os.path.join(root_dir, "saved_models")

# Create directories if not present
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)



# ============================================================================
#  HELPER: Decoder channels must match encoder depth
# ============================================================================
def get_decoder_channels(depth):
    """
    SMP-UNet requires decoder_channels list length = encoder_depth.
    Example:
        depth=5 → [256,128,64,32,16]
        depth=4 → [256,128,64,32]
    """
    base = [256, 128, 64, 32, 16]

    if depth not in [3, 4, 5]:
        raise ValueError("encoder_depth must be 3, 4, or 5")

    return base[:depth]



# ============================================================================
#  BUILD SMP UNET MODEL
# ============================================================================
def build_model(encoder, pretrained, encoder_depth, num_classes):
    """
    Creates an SMP-UNet model with the given backbone, pretrained flag,
    encoder depth, and number of segmentation classes.

    Args:
        encoder (str): Encoder backbone name (resnet101, mobilenet_v2, etc.)
        pretrained (bool): Use ImageNet weights or not.
        encoder_depth (int): Depth of encoder (3–5).
        num_classes (int): Output segmentation classes.

    Returns:
        smp.Unet: Constructed PyTorch UNet model.
    """

    decoder_channels = get_decoder_channels(encoder_depth)

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
        classes=num_classes,
        activation=None,
    )

    return model



# ============================================================================
#  DEFAULT TRAINING (NO ABLATION)
# ============================================================================
def run_default(args):
    """
    Standard training pipeline without any ablation.
    Uses user-specified encoder, pretrained flag, encoder depth,
    number of classes, learning rate, epochs, etc.

    Saves best model as <model_name>_model.pt
    """

    print("\n==============================")
    print(" Running DEFAULT TRAINING...")
    print("==============================\n")
    args.pretrained = not args.pretrained  # Toggle for default run
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------
    # Load dataset
    # -------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        "data-urban", args.batch_size
    )

    # -------------------------------
    # Build model
    # -------------------------------
    model = build_model(
        encoder=args.encoder,
        pretrained=args.pretrained,
        encoder_depth=args.encoder_depth,
        num_classes=args.num_classes,
    ).to(device)

    # -------------------------------
    # Loss, optimizer, scheduler
    # -------------------------------
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )

    # -------------------------------
    # Train
    # -------------------------------
    history = fit(
        epochs=args.epochs,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_classes=args.num_classes,
        model_name=args.model_name,
        saved_path=models_dir,
        device=device,
    )

    # -------------------------------
    # Load best checkpoint
    # -------------------------------
    ckpt = os.path.join(models_dir, f"{args.model_name}_model.pt")

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=device))

    # -------------------------------
    # Test Evaluation
    # -------------------------------
    test_metrics = evaluate_on_test(
        model=model,
        test_loader=test_loader,
        num_classes=args.num_classes,
        device=device,
    )

    print("\n>>> FINAL TEST METRICS <<<")
    print(test_metrics)



# ============================================================================
#  ABLATION STUDY
# ============================================================================
def run_ablation(args):
    """
    Runs ablation over one variable at a time:
        - encoder
        - pretrained
        - encoder_depth

    The other parameters remain fixed as set by argparse.
    Saves a results_<mode>.xlsx file.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        "data-urban", args.batch_size
    )

    # ----------------------------------------------
    # Select ablation parameter
    # ----------------------------------------------
    if args.mode == "encoder":
        search_space = {"param": "encoder", "values": ["resnet101", "mobilenet_v2", "efficientnet-b3"]}

    elif args.mode == "pretrained":
        search_space = {"param": "pretrained", "values": [True, False]}

    elif args.mode == "encoder_depth":
        search_space = {"param": "encoder_depth", "values": [3, 4, 5]}

    else:
        raise ValueError("Mode must be: default | encoder | pretrained | encoder_depth")

    print(f"\n===== RUNNING ABLATION ON {args.mode.upper()} =====")

    results = []  # store results across all runs

    # ----------------------------------------------
    # Loop over ablation settings
    # ----------------------------------------------
    for val in search_space["values"]:

        print("\n--------------------------------------------")
        print(f"Running {search_space['param']} = {val}")
        print("--------------------------------------------")

        # Copy defaults
        encoder = args.encoder
        pretrained = args.pretrained
        encoder_depth = args.encoder_depth

        # Override only the variable being ablated
        if args.mode == "encoder":
            encoder = val
        elif args.mode == "pretrained":
            pretrained = val
        elif args.mode == "encoder_depth":
            encoder_depth = val

        # Build model
        model = build_model(
            encoder=encoder,
            pretrained=pretrained,
            encoder_depth=encoder_depth,
            num_classes=args.num_classes,
        ).to(device)

        # Loss + optimizer + schedule
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
        )

        model_name = f"ablation_{args.mode}_{val}"

        # Train and save best
        history = fit(
            epochs=args.epochs,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_classes=args.num_classes,
            model_name=model_name,
            saved_path=models_dir,
            device=device,
        )

        ckpt = os.path.join(models_dir, f"{model_name}_model.pt")
        print(f"Loading checkpoint: {ckpt}")

        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Model not found: {ckpt}")

        model.load_state_dict(torch.load(ckpt, map_location=device))

        # Test evaluation
        test_metrics = evaluate_on_test(
            model=model,
            test_loader=test_loader,
            num_classes=args.num_classes,
            device=device,
        )

        # Store summary
        results.append({
            search_space["param"]: val,
            "best_val_loss": min(history["val_loss"]),
            "best_val_miou": max(history["val_miou"]),
            "best_val_dice": max(history["val_dice"]),
            "test_miou": test_metrics["test_miou"],
            "test_dice": test_metrics["test_dice"],
            "test_acc": test_metrics["test_acc"],
        })

    # Save results
    out_file = os.path.join(root_dir,f"results_{args.mode}.xlsx")
    pd.DataFrame(results).to_excel(out_file, index=False)

    print(f"\nAblation Results saved to → {out_file}")



# ============================================================================
#  MAIN FUNCTION
# ============================================================================
def main():
    """
    CLI for:
        default training
        ablation (encoder | pretrained | encoder_depth)
    """
    parser = argparse.ArgumentParser()

    # Main mode
    parser.add_argument("--mode", type=str, default="default",
                        help="default | encoder | pretrained | encoder_depth")

    # UNet settings
    parser.add_argument("--encoder", type=str, default="resnet101")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--encoder_depth", type=int, default=5)

    # Training settings
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--model_name", type=str, default="unet_resnet101")

    args = parser.parse_args()

    if args.mode == "default":
        run_default(args)
    else:
        run_ablation(args)


if __name__ == "__main__":
    main()
