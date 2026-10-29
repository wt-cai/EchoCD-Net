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

## Citation
if you find our work useful, please cite our paper, thank you!
```bash
@article{cai2026segmotion,
  title={SegMotion-Net: Segmentation-guided motion analysis for early myocardial infarction detection from echocardiography video},
  author={Cai, Weitao and Ren, Hao and Jing, Fengshi and Xu, Zhongzhi and Zhou, Jiandong and Liu, Peng and Sun, Yu and Jin, Wen and Ye, Kunlin and Qin, Danmin and others},
  journal={Medical Image Analysis},
  pages={104212},
  year={2026},
  publisher={Elsevier}
}
```
