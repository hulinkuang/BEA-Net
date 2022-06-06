# BEG-SegNet
The code for the BIBM2021 paper "BEA-SegNet: Body and Edge Aware Network for Medical Image Segmentation" and the extension paper submitted to IEEE TMI "BEG-SegNet: Medical Image Segmentation Network with Body and Edge Generation"
## Requirements
CUDA 11.4<br />
Python 3.8.12<br /> 
Pytorch 1.11.0<br />
Torchvision 0.12.0<br />
batchgenerators 0.21<br />
SimpleITK 2.1.1 <br />
scipy 1.8.0 <br />
kornia 0.6.4 <br />

## Usage

### 0. Installation
* Install nnUNet and BSG as below
  
```
cd nnUNet
pip install -e .

cd BSG_package
pip install -e .
```

### 1. Training 
cd BSG_package/BSG/run

* Run `python run_training.py -gpu='0' -outpath='BSG'` for training.

### 2. Testing 
* Run `python run_training.py -gpu='0' -outpath='BSG' -val --val_folder='validation_output'` for validation.

# Datasets
ISIC2018 dataset for skin lesion segmentation from (https://challenge.isic-archive.com/data/#2018). There is an example to process raw data in ./BSG-package/BSG/demo/turial.py.

For other datasets, the preprocess files can be found at nnUNet/nnunet/dataset_conversion. All pre-trained model can be download from 链接：https://pan.baidu.com/s/1o9pKTCzsJW6CzCxFTZ8DTg 
提取码：7uvl


### Acknowledgements
Part of codes are reused from the nnU-Net. Thanks to Fabian Isensee for the codes of nnU-Net.


