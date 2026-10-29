# EchoCD-Net
EchoCD-Net: Cross-Domain Left Ventricular Segmentation in Echocardiographic Video

## Installation

```bash
conda create --name echocd python=3.10
conda activate echocd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Download the pretrained SAM ViT-B checkpoint and place it in the repository root

## Usage
### Training

```bash
python train.py
```

### Testing

```bash
python test.py
```

## Acknowledgement

The work is based on MemSAM, and XMem. Thanks for the open source contributions to these efforts!
