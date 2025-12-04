import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# ----------------------------------------------------
# Global root directory (current working directory)
# ----------------------------------------------------
# Get the directory where THIS script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up → project root
root = os.path.dirname(script_dir)



# ============================================================
# SegmentationDataset
# ============================================================
class SegmentationDataset(Dataset):
    """
    Custom dataset for semantic segmentation.
    """

    def __init__(self, image_paths: list, mask_paths: list, train: bool):
        """
        Args:
            image_paths (list): List of file paths to input images.
            mask_paths  (list): List of file paths to segmentation masks.
            train (bool): If True, apply augmentations.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.train = train

        # Safety check
        assert len(self.image_paths) == len(self.mask_paths), \
            "Number of images and masks must match!"

    def __len__(self):
        """Returns the dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads one (image, mask) example.
        Returns:
            (Tensor, Tensor): (image_tensor, mask_tensor)
        """

        # ----------------------------------------------------
        # Load image and mask files
        # ----------------------------------------------------
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        # Convert image float tensor in [0,1]
        img_tensor = TF.to_tensor(img)

        # Convert mask LongTensor with class IDs
        mask_tensor = torch.from_numpy(np.array(mask)).long()

        # Apply augmentation if training
        if self.train:
            img_tensor, mask_tensor = self._transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor

    

# ============================================================
# DataLoader Factory
# ============================================================
def get_dataloaders(path: str, batch_size: int):
    """
    Constructs DataLoaders for training, validation, and testing.

    Args:
        path (str): Path under root where dataset folders exist.
                    Example: "data-urban"
        batch_size (int): Batch size for loaders.

    Returns:
        train_loader, val_loader, test_loader
    """

    # Build full directory paths
    train_imgs  = sorted(glob.glob(os.path.join(root, path, "train_images/*")))
    train_masks = sorted(glob.glob(os.path.join(root, path, "train_masks/*")))

    val_imgs    = sorted(glob.glob(os.path.join(root, path, "val_images/*")))
    val_masks   = sorted(glob.glob(os.path.join(root, path, "val_masks/*")))

    test_imgs   = sorted(glob.glob(os.path.join(root, path, "test_images/*")))
    test_masks  = sorted(glob.glob(os.path.join(root, path, "test_masks/*")))

    # Create Dataset objects
    train_set = SegmentationDataset(train_imgs, train_masks, train=True)
    val_set   = SegmentationDataset(val_imgs, val_masks, train=False)
    test_set  = SegmentationDataset(test_imgs, test_masks, train=False)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader   = DataLoader(val_set, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    test_loader  = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
