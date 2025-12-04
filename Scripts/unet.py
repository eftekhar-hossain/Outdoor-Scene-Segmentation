import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# =====================================================================
#   Double Convolution Block
#   (Conv - BatchNorm - ReLU) × 2
# =====================================================================
class DoubleConv(nn.Module):
    """
    A stack of two 3×3 convolutions each followed by BatchNorm + ReLU.
    This is used repeatedly in UNet for both encoder and decoder stages.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            # First 3×3 convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Second 3×3 convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Apply the sequential block
        return self.block(x)



# =====================================================================
#                               UNET
# =====================================================================
class UNET(nn.Module):
    """
    A classic UNet for semantic segmentation.
    
    Structure:
        Encoder (Down path):  Repeated (DoubleConv -> MaxPool)
        Bottleneck:           DoubleConv at smallest resolution
        Decoder (Up path):    TransposeConv -> Concatenate skip -> DoubleConv
    """

    def __init__(self, in_channels=3, out_channels=12,
                 features=[16, 32, 64, 128, 256, 512]):
        super().__init__()

        # Store encoder and decoder blocks
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()

        # MaxPool used for downsampling after each encoder stage
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --------------------------------------------------------------
        # ENCODER (Downsampling path)
        # --------------------------------------------------------------
        # Example: 3 -> 16, 16->32, 32->64, ...
        current_channels = in_channels

        for feat in features:
            # Apply a DoubleConv block
            self.downs.append(DoubleConv(current_channels, feat))

            # Update the number of channels flowing forward
            current_channels = feat

        # --------------------------------------------------------------
        # BOTTLENECK (deepest part of UNet)
        # --------------------------------------------------------------
        # Example: 512 -> 1024 -> 1024
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # --------------------------------------------------------------
        # DECODER (Up path)
        # --------------------------------------------------------------
        # For each level, we:
        #   1) Upsample using ConvTranspose
        #   2) Concatenate the skip connection from encoder
        #   3) Apply DoubleConv
        for feat in reversed(features):

            # Step 1: Learnable upsampling (transposed convolution)
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feat * 2,  # because bottleneck outputs 2× channels
                    out_channels=feat,
                    kernel_size=2,
                    stride=2
                )
            )

            # Step 2: Double convolution after concatenation
            self.ups.append(
                DoubleConv(
                    in_channels=feat * 2,  # skip + upsampled feature maps
                    out_channels=feat
                )
            )

        # --------------------------------------------------------------
        # FINAL 1×1 CONV
        # Maps feature maps -> segmentation logits per class
        # --------------------------------------------------------------
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    # =================================================================
    #                           Forward Pass
    # =================================================================
    def forward(self, x):
        """
        Forward pass through UNet:
            - Go down the encoder, store skip connections
            - Pass through bottleneck
            - Go up the decoder using skip connections
        """
        
        skip_connections = []

        # ---------------- ENCODER ----------------
        for down in self.downs:
            x = down(x)                    # DoubleConv block
            skip_connections.append(x)     # Store for later concatenation
            x = self.pool(x)               # Downsample by factor 2

        # ---------------- BOTTLENECK ----------------
        x = self.bottleneck(x)

        # Reverse skip connections so deepest comes first in decoding
        skip_connections = skip_connections[::-1]

        # ---------------- DECODER ----------------
        # The self.ups list is arranged as: [UpConv, DoubleConv, UpConv, DoubleConv, ...]
        for idx in range(0, len(self.ups), 2):

            # Step 1: Transposed Conv -> upsample
            x = self.ups[idx](x)

            # Step 2: Retrieve matching skip connection
            skip = skip_connections[idx // 2]

            # Step 3 (rare): Fix shape mismatch due to rounding in odd-sized inputs
            if x.shape[2:] != skip.shape[2:]:
                x = TF.resize(x, size=skip.shape[2:])

            # Step 4: Concatenate skip + upsampled features along channel dimension
            x = torch.cat((skip, x), dim=1)

            # Step 5: Apply DoubleConv
            x = self.ups[idx + 1](x)

        # ---------------- CLASSIFICATION LAYER ----------------
        return self.final_conv(x)
