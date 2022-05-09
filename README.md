# BEG-SegNet
The code for the BIBM2021 paper "BEA-SegNet: Body and Edge Aware Network for Medical Image Segmentation" and the extension paper submitted to IEEE TMI "BEG-SegNet: Medical Image Segmentation Network with Body and Edge Generation"
# requirments
python=3.5.6 

tensorflow-gpu=1.9 

keras-base=2.2.2 

medpy

openCV

# Datasets
ISIC2018 dataset for skin lesion segmentation from (https://challenge.isic-archive.com/data/#2018). put all RGB images and its lesion GT images into a folder "./dataset_isic18/" . run the Prepare_data_ISIC2018.py to prepare the needed .npy files for training and testing

# run train, test and evaluation
and then train_BEG_SegNet.py to train the model, and run evaluate1_BEG_SEGNET.py to evluate the trained model on the test set and compute evaluation metrics.
