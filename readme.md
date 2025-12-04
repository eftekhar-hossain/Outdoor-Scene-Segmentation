# Semantic Segmentation of Outdoor Scenes using UNet Variants 

This repository provides a complete and reproducible pipeline for **semantic segmentation** on outdoor scene images using various UNet architectures. The focus is on segmenting key elements such as roads, buildings, vegetation, and sky.

This project is done under **Course CAP-5415 - Computer Vision at University of Central Florida, USA.**


This README explains:
- Repository structure  
- How to install & set up environment  
- How to train UNet / Attention UNet  
- How to run ablations  
- Expected outputs 



## Setup Instructions

Follow these steps to set up the environment and download the dataset.


### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```
### 2. Create a Virtual Environment

```bash
conda a create -n cap5415 python=3.10
conda activate cap5415
```
### 3. Install Dataset and Dependencies

```bash
bash setup.sh
```
### 4. Repository Structure

The repository will look like as follows: 
```
Outdoor-Scene-Segmentation/
│
├── Scripts/
│   ├── dataloader.py          # Dataset & DataLoader logic
│   ├── trainer.py             # Training + validation + evaluation
│   ├── metrics.py             # mIoU, Dice-F1, Pixel Accuracy
│   ├── utils.py               # Training curve plotting
│   ├── unet.py                # Standard UNet implementation
│   ├── att_unet.py            # Attention UNet implementation
│   ├── train_unet.py          # Train UNet or Attention UNet
│   ├── ablation.py            # Encoder / depth / pretrained ablations
│
├── data-urban/                # Dataset (created after setup)
│   ├── train_images/
│   ├── train_masks/
│   ├── val_images/
│   ├── val_masks/
│   ├── test_images/
│   ├── test_masks/
│
├── saved_models/              # Best checkpoints stored here
├── plots/                     # Training curve PNGs saved here
│
├── setup.sh                   # Dataset download + environment setup
├── requirements.txt
└── README.md
```


## Model Training

Make sure you are in the ```Scripts/``` directory to run the training scripts.
    ```
    cd Scripts
    ```

- Train Standard UNet
    ```bash
    python train_unet.py --model_type unet
    ```
- Train Attention UNet
    ```bash
    python rain_unet.py --model_type attunet
    ```

### CLI Arguments (train_unet.py)

| Argument       | Description              | Default     |
|----------------|--------------------------|-------------|
| `--model_type` | `unet` or `attunet`      | `unet`      |
| `--data_root`  | dataset directory        | `data-urban`|
| `--epochs`     | training epochs          | `50`        |
| `--batch_size` | batch size               | `8`         |
| `--lr`         | learning rate            | `1e-3`      |
| `--num_classes`| segmentation classes     | `12`        |


---

### Output Files

| File | Description |
|------|-------------|
| `saved_models/unet_model.pt` | Best UNet checkpoint |
| `saved_models/attunet_model.pt` | Best Attention UNet checkpoint |
| `training_history_unet.xlsx` | Loss & metric logs |
| `plots/training_history_unet.png` | Training curves |
| `plots/training_history_attunet.png` | Training curves |


---

## Ablation Experiments (SMP-UNet)

Ablations include:

### **1.  Encoder backbone**
- `resnet101`
- `mobilenet_v2`
- `efficientnet-b3`

### **2. Pretrained weights**
- `True`
- `False`

### **3. Encoder depth**
- `3`, `4`, `5`


---

##  Run Unet with ResNet101

```bash
python ablation.py --mode default
```

## Run Encoder Ablation

```bash
python ablation.py --mode encoder
```
## Run Pretrained Weights Ablation

```bash
python ablation.py --mode pretrained
```
## Run Encoder Depth Ablation

```bash
python ablation.py --mode encoder_depth
```
### CLI Arguments (ablation.py)
| Argument        | Description                          | Default        |
|-----------------|--------------------------------------|----------------|
| `--mode`        | `default`, `encoder`, `pretrained`, `depth` | `default`      |
| `--data_root`   | dataset directory                    | `data-urban`     |
| `--epochs`      | training epochs                      | `50`          |
| `--batch_size`  | batch size                           | `8`            |
| `--lr`          | learning rate                        | `1e-3`        |
| `--num_classes` | segmentation classes                 | `12`          |  
---

## Acknowledgements
- For understanding the UNet architecture we took help from the kaggle notebook: https://www.kaggle.com/datasets/kooaslansefat/uav-segmentation-aeroscapes/data
