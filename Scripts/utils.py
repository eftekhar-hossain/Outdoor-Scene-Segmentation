import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================================================
#               PLOT TRAINING CURVES (4-SUBPLOT)
# ===============================================================
def plot_training_history(history,model_name,save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(22, 4))

    # ---- LOSS ----
    plt.subplot(1, 4, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # ---- ACCURACY ----
    plt.subplot(1, 4, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    # ---- DICE ----
    plt.subplot(1, 4, 3)
    plt.plot(epochs, history["train_dice"], label="Train Dice")
    plt.plot(epochs, history["val_dice"], label="Val Dice")
    plt.title("Dice / F1 Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.grid(True)
    plt.legend()

    # ---- MIOU ----
    plt.subplot(1, 4, 4)
    plt.plot(epochs, history["train_miou"], label="Train mIoU")
    plt.plot(epochs, history["val_miou"], label="Val mIoU")
    plt.title("mIoU Curve")
    plt.xlabel("Epochs")
    plt.ylabel("mIoU")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_history_{model_name}.png"), dpi=600)
    plt.show()
    print(f"Training history plot saved as training_history_{model_name}.png")