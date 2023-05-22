# BEA-Net
The code for the BIBM2021 paper "BEA-SegNet: Body and Edge Aware Network for Medical Image Segmentation" and the extension paper "BEA-Net: Body and Edge Aware Network with Multi-Scale Short-Term Concatenation for Medical Image Segmentation" <br />
This paper proposes a new network with body and edge generation modules, multi-scale short-term concatenation, parallel body and edge decoders and their fusion, and body and edge supervision. It outperform several state-of-the-arts on six datasets for six different medical image segmentation tasks.
![framework](https://user-images.githubusercontent.com/35280235/172337360-34553fff-ed5c-4e72-90e0-f92bceff9d37.jpg)


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
### 1 Skin lesion segmentation on ISIC2018 dataset
### 1.1 Dataset access and pre-processing
ISIC2018 dataset for skin lesion segmentation can be downloaded from (https://challenge.isic-archive.com/data/#2018). Pre-process the datasets using the preprocess codes in nnUNet/nnunet/dataset_conversion.
### 1.2 Training

* Run `BEA_train -gpu='0' -task={task_id} -outpath='BEA'` for training.

### 1.3 Testing 
* Run `BEA_train -gpu='0' -task={task_id} -outpath='BEA' -val --val_folder='validation_output'` for testing. 

### 1.4 Testing demo with pre-trained model
The pre-trained model using ISIC2018 can be download from [[Baidu YUN]](https://pan.baidu.com/s/1EsS5L7GJfND6E2XLX_H_Pw?pwd=xfi5) with the password "xfi5". We give a testing demo which can segment raw images with skin lesions and is implemented via running ./BEA-package/BEA/demo/turial.py.

### 2 Medical segmentation on other five datasets
### Dataset access
Kvasir-SEG: https://datasets.simula.no/kvasir-seg/  <br />
IDRiD: https://idrid.grand-challenge.org/  <br />
BUSI: https://scholar.cu.edu.eg/?q=afahmy/pages/dataset   <br />
CVC-ClincDB: https://polyp.grand-challenge.org/CVCClinicDB/  <br />
JSRT: http://db.jsrt.or.jp/eng.php <br />

### 2.1 pre-processing, training and testing
To pre-process the datasets, please use the corresponding preprocess codes in nnUNet/nnunet/dataset_conversion. the training and testing for other datasets are similar.

## Acknowledgements
Part of codes are reused from the nnU-Net. Thanks to Fabian Isensee for the codes of nnU-Net.

## Citations
if you use this code, please cite the following paper:
[1] Hulin Kuang, Yixiong Liang, Ning Liu, Jin Liu, Jianxin Wang, “BEA-SegNet: Body and Edge Aware Network for Medical Image Segmentation”, in Proc. of 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2021, pp. 939-944. 