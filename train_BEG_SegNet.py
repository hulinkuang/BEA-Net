# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019

@author: Reza Azad
"""
from __future__ import division
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import BEG_SegNet as M
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import callbacks
import pickle
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler, EarlyStopping


# ===== normalize over the dataset
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                    np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
    return imgs_normalized


####################################  Load Data #####################################
tr_data = np.load('data_train.npy')
te_data = np.load('data_test.npy')
val_data = np.load('data_val.npy')

tr_mask = np.load('mask_train.npy')
te_mask = np.load('mask_test.npy')
val_mask = np.load('mask_val.npy')

tr_mask = np.expand_dims(tr_mask, axis=3)
te_mask = np.expand_dims(te_mask, axis=3)
val_mask = np.expand_dims(val_mask, axis=3)

######### edg ###########
tr_mask_edg = np.load('mask_train_edg.npy')
te_mask_edg = np.load('mask_test_edg.npy')
val_mask_edg = np.load('mask_val_edg.npy')

tr_mask_edg = np.expand_dims(tr_mask_edg, axis=3)
te_mask_edg = np.expand_dims(te_mask_edg, axis=3)
val_mask_edg = np.expand_dims(val_mask_edg, axis=3)

print('ISIC18 Dataset loaded')

tr_data = dataset_normalized(tr_data)
te_data = dataset_normalized(te_data)
val_data = dataset_normalized(val_data)

tr_mask = tr_mask / 255.
te_mask = te_mask / 255.
val_mask = val_mask / 255.

tr_mask_edg = tr_mask_edg/255.
te_mask_edg = te_mask_edg/255.
val_mask_edg = val_mask_edg/255.


print('dataset Normalized')

# Build model
model = M.unet(input_size=(256, 256, 3))
# model.summary()

print('Training')
batch_size = 8
nb_epoch = 200

mcp_save = ModelCheckpoint('trained_BEG_SegNet_model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, verbose=1, patience=25)

history = model.fit(tr_data, [tr_mask, tr_mask_edg],
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    shuffle=True,
                    verbose=1,
                    validation_data=(val_data, [val_mask, val_mask_edg]),
                    callbacks=[mcp_save, reduce_lr_loss, earlystop])

print('Trained model saved')
# with open('hist_isic18_1207', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)



