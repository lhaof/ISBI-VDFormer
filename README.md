# VIEW-DISENTANGLED TRANSFORMER FOR BRAIN LESION DETECTION-ISBI 2022

This repo contains the supported code and configuration files of [View-Disentangled Transformer](https://empty.com). It is based on [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).

## Pretrain Model

|   Model   |                            config                            |                            Params                            |
| :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Baseline  | [config](VDFormer/configs/VDFormer/SwinCascadeRCNN_config.py) | [google drive](https://drive.google.com/file/d/1_kn-7yX6vD5YPB4l61lAX6IgIBR1yPgW/view?usp=sharing) |
| +VDFormer | [config](VDFormer/configs/VDFormer/SwinCascadeRCNN_VDFormer_config.py) | [google drive](https://drive.google.com/file/d/1xsIn5gauJNBf38BhQBV2Kvd8fJCzpGG7/view?usp=sharing) |

## Prerequisites

- Linux
- Python 3.7.11
- Pytorch 1.9.1
- CUDA 10.2
- MMDetection 2.11.0
- MMCV 1.3.14

## Installation

### Prepare environment

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n VDFormer python=3.7 -y
    conda activate VDFormer
    ```
2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch cudatoolkit=10.2 torchvision -c pytorch
    ```
### Install MMDetection
1. Install mmcv-full
    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu102}/{torch1.9.0}/index.html
    ```
2. Install MMDetection-VDFormer
   ```shell
   git clone $https://github.com/VDFormer(The github link of VD Former)
   cd $VDFormer(The folder of VD Former)
   pip install -r requirements/build.txt
   python setup.py develop
   ```
### Dataset
We use dataset in coco format. 

If you want to train VD-Former on other data, you need to change the format to coco and the data path in `config/VDFormer/base_config.py`

### Model definition
The VDFormer model is mainly defined in `mmdet/models/swin_transformer_proposed.py` and `mmdet/models/necks/swin_fusion_layer_proposed.py`
### Inference
```python
# sinlge-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox
```
### Training
```python
# sinlge-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL>
# multi-gpu training
CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh <CONFIG_FILE> 2 --cfg-options model.pretrained=<PRETRAIN_MODEL>
```
## Citing VD Transformer
```
@artical{li2022VDFormer,
    title={VIEW-DISENTANGLED TRANSFORMER FOR BRAIN LESION DETECTION},
    year={2022}
}
```
## Contact
```
2022.3.12 
Junjia Huang
1959643995@qq.com
```
