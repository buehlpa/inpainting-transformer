# inpainting-transformer
This is unofficial pytorch implementation of Inpainting Transformer for Anomaly Detection

We support train and test(slow version) for InTra models now. Updates would be coming soon.

## Acknowledgement
This code is based on the following repositories:
https://github.com/taikiinoue45/RIAD

## Prerequisites
- Python 3
  - torch
  - opencv
  - PIL
  - albumentations

## Usage

### Train
```bash
python main.py --image_dir=../mvtec_anomaly_detection/bottle/ --ckpt=./ckpt/InTra/MVTAD_bottle/
```

### Inference
```bash
python main.py --image_dir=../mvtec_anomaly_detection/bottle/ --ckpt=./ckpt/InTra/MVTAD_bottle/ --is_infer
```


### own
```bash
#inference
python main.py --image_dir=/home/bule/projects/MVTec_Visualizer/data/mvtec_anomaly_detection/cable --ckpt=/home/bule/projects/inpainting-transformer/ckpt --is_infer
```


## TODO(coming soon!)
Code refactoring

speed up inference

Add pretrained models

update prerequisites

update documents



### Issue

- missing evalatuion part , cannnot reproduce results from paper
- 