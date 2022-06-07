# BEA-SegNet
The code for the BIBM2021 paper "BEA-SegNet: Body and Edge Aware Network for Medical Image Segmentation" and the extension paper submitted to IEEE TMI "BEA-SegNet: Medical Image Segmentation Network with Body and Edge Generation"
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
* Install nnUNet and BEA as below
  
```
cd nnUNet
pip install -e .

cd BEA_package
pip install -e .
```

### 1. Training

* Run `BEA_train -gpu='0' -task={task_id} -outpath='BEA'` for training.

### 2. Testing 
* Run `BEA_train -gpu='0' -task={task_id} -outpath='BEA' -val --val_folder='validation_output'` for validation.

# Datasets
ISIC2018 dataset for skin lesion segmentation from (https://challenge.isic-archive.com/data/#2018). There is an example to process raw data in ./BEA-package/BEA/demo/turial.py.

For other datasets, the preprocess files can be found at nnUNet/nnunet/dataset_conversion. All pre-trained model can be download from [[Baidu YUN]](https://pan.baidu.com/s/1o9pKTCzsJW6CzCxFTZ8DTg) with the password "7uvl".

### Acknowledgements
Part of codes are reused from the nnU-Net. Thanks to Fabian Isensee for the codes of nnU-Net.


