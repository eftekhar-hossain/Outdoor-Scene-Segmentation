#!/bin/bash

echo "[INFO] Starting setup..."

# -----------------------------
# Install required tools
# -----------------------------
pip install gdown --quiet

# -----------------------------
# Download dataset ZIP
# -----------------------------
DATA_URL="https://drive.google.com/uc?id=1uv-qIYZQNm51o-uU4NcfFq9bmYRiwOXd"
ZIP_NAME="dataset.zip"

echo "[INFO] Downloading dataset..."
gdown "$DATA_URL" -O "$ZIP_NAME"

# -----------------------------
# Extract ZIP to temp folder
# -----------------------------
TEMP_DIR="temp_dataset_extract"
mkdir -p "$TEMP_DIR"

echo "[INFO] Extracting ZIP..."
unzip -q "$ZIP_NAME" -d "$TEMP_DIR"

# -----------------------------
# Locate inner data-urban directory
# -----------------------------
echo "[INFO] Searching for inner 'data-urban' folder..."

INNER_PATH=$(find "$TEMP_DIR" -type d -name "data-urban" | head -n 1)

if [ -z "$INNER_PATH" ]; then
    echo "[ERROR] No data-urban folder found inside ZIP!"
    exit 1
fi

echo "[INFO] Found dataset folder at: $INNER_PATH"

# -----------------------------
# Move final dataset to root-level data-urban/
# -----------------------------
echo "[INFO] Moving dataset to ./data-urban..."

rm -rf data-urban
mkdir -p data-urban

# Move contents (not the folder itself)
mv "$INNER_PATH"/* data-urban/

# -----------------------------
# Cleanup
# -----------------------------
echo "[INFO] Cleaning up..."
rm -rf "$TEMP_DIR"
rm "$ZIP_NAME"

# -----------------------------
# Install Python requirements
# -----------------------------
echo "[INFO] Installing requirements..."
pip install -r requirements.txt

echo "[INFO] Setup complete! Dataset is ready in ./data-urban/"
