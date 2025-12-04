import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# ==============================================================
# 1. DOUBLE CONVOLUTION BLOCK (Conv -> BN -> ReLU) × 2
# ==============================================================
class DoubleConvNew(nn.Module):
    """
    Applies two convolution blocks:
        Conv -> BatchNorm -> ReLU
        Conv -> BatchNorm -> ReLU
    Used in both Encoder and Decoder parts.
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConvNew, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ==============================================================
# 2. ATTENTION GATE (AG)
# ==============================================================
class AttentionGate(nn.Module):
    """
    Standard Attention Gate from 'Attention U-Net' paper.
    
    Args:
        F_g  : Channels of gating signal (from decoder)
        F_l  : Channels of skip-connection (from encoder)
        F_int: Number of internal feature channels

    The AG produces a mask (0–1) to suppress irrelevant encoder features.
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        # 1x1 conv to match dimensions of gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        # 1x1 conv to match dimensions of skip features
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        # Final attention mask generation
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()         # output ∈ [0,1]
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        """
        x : skip-connection features (from encoder)
        g : gating signal (from decoder)
        """

        # Linear projections for gating & skip tensors
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Combine & activate
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)      # attention weights

        # Multiply mask with skip feature map
        return x * psi


# ==============================================================
# 3. ATTENTION U-NET
# ==============================================================
class AttentionUNet(nn.Module):
    """
    Full Attention U-Net architecture with learned skip attention.
    The model enhances standard U-Net by suppressing irrelevant
    encoder features using Attention Gates (AGs).

    Args:
        in_channels (int): Input channels (3 for RGB)
        out_channels (int): Number of segmentation classes
    """

    def __init__(self, in_channels=3, out_channels=12):
        super(AttentionUNet, self).__init__()

        # =============================
        # Encoder (Downsampling Path)
        # =============================
        self.conv1 = DoubleConvNew(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConvNew(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConvNew(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConvNew(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv5 = DoubleConvNew(512, 1024)

        # =============================
        # Attention Gates (AGs)
        # =============================
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionGate(F_g=64,  F_l=64,  F_int=32)

        # =============================
        # Decoder (Upsampling Path)
        # =============================
        self.up4      = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConvNew(1024, 512)

        self.up3      = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConvNew(512, 256)

        self.up2      = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConvNew(256, 128)

        self.up1      = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConvNew(128, 64)

        # Final segmentation layer (no activation — use softmax in loss later)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)


    # ==========================================================
    #                   FORWARD PASS
    # ==========================================================
    def forward(self, x):

        # ----- Encoder -----
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        bn = self.conv5(p4)

        # ----- Decoder + Attention Gates -----

        # Level 4
        up4 = self.up4(bn)
        att4 = self.att4(c4, up4)
        up4 = torch.cat([att4, up4], dim=1)
        up4 = self.up_conv4(up4)

        # Level 3
        up3 = self.up3(up4)
        att3 = self.att3(c3, up3)
        up3 = torch.cat([att3, up3], dim=1)
        up3 = self.up_conv3(up3)

        # Level 2
        up2 = self.up2(up3)
        att2 = self.att2(c2, up2)
        up2 = torch.cat([att2, up2], dim=1)
        up2 = self.up_conv2(up2)

        # Level 1
        up1 = self.up1(up2)
        att1 = self.att1(c1, up1)
        up1 = torch.cat([att1, up1], dim=1)
        up1 = self.up_conv1(up1)

        # Final classifier
        return self.final(up1)
