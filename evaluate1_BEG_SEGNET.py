from __future__ import division
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import BEG_SegNet as M
import numpy as np
# import scipy
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from functools import partial
import scipy.io as scio
from keras import backend as K
import tensorflow as tf
import scipy.misc
from scipy import ndimage


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_coef1(y_true, y_pred):
    y_true_f = y_true.astype(float)
    y_pred_f = y_pred.astype(float)
    # y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * np.sum(intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))
    return score


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
te_data = np.load('data_test.npy')
te_mask = np.load('mask_test.npy')
te_mask = np.expand_dims(te_mask, axis=3)
te_mask_edg = np.load('mask_test_edg.npy')
te_mask_edg = np.expand_dims(te_mask_edg, axis=3)

print('ISIC18 Dataset loaded')

te_data = dataset_normalized(te_data)

model = M.unet(input_size=(256, 256, 3))
model.load_weights('trained_BEG_SegNet_model.h5')

[predictions, edg, body, final_feat] = model.predict(te_data, batch_size=8, verbose=1)
output_folder = 'output/'

from medpy import metric

N = 540
dice = np.zeros([N])
hd95 = np.zeros([N])
sen = np.zeros([N])
spe = np.zeros([N])
acc = np.zeros([N])
pre = np.zeros([N])
jac = np.zeros([N])
para = np.zeros([N, 7])
th = 0.5
res_dir = 'Results_Test_edge_visual/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
for idx in range(0, N):

    print(idx)
    y_true = np.squeeze(te_mask[idx])
    y_true = np.where(y_true >= 0.5, 1, 0)
    y_scores = np.squeeze(predictions[idx])

    y_pred = np.where(y_scores >= th, 1, 0)

    if np.max(y_pred) == 0:
        hd0 = 10000
    else:
        hd0 = metric.binary.hd95(y_pred, y_true)

    sens0 = metric.binary.sensitivity(y_pred, y_true)
    dice0 = metric.binary.dc(y_pred, y_true)
    spe0 = metric.binary.specificity(y_pred, y_true)
    pre0 = metric.binary.precision(y_pred, y_true)
    jc0 = metric.binary.jc(y_pred, y_true)
    if jc0 < 0.65:
        jc0 = 0
    jac[idx] = jc0
    eid = np.where((y_pred - y_true)== 0)
    acc0 = len(y_pred[eid])/(256*256)
    acc[idx] = acc0
    dice[idx] = dice0
    hd95[idx] = hd0
    spe[idx] = spe0
    sen[idx] = sens0
    pre[idx] = pre0

    te_img = np.squeeze(te_data[idx])
    scipy.misc.imsave(res_dir + str(idx) + '_img.png', te_img)
    te_gt = np.squeeze(te_mask[idx])
    scipy.misc.imsave(res_dir + str(idx) + '_gt.png', te_gt)
    te_gt_edg = np.squeeze(te_mask_edg[idx])
    scipy.misc.imsave(res_dir + str(idx) + '_gt_edge.png', te_gt_edg)


    scipy.misc.imsave(res_dir + str(idx) + '_final_prob.png', y_scores)
    scipy.misc.imsave(res_dir + str(idx) + '_final_seg.png', y_pred)

    edg_scores = np.squeeze(edg[idx])
    # tmp0 = edg_scores.reshape(256 * 256,1)
    # edg_scores = (edg_scores - np.min(tmp0)) / (np.max(tmp0)-np.min(tmp0))
    body_scores = np.squeeze(body[idx])
    edg_pred = np.where(edg_scores >= 0.15, 1, 0)
    body_pred = np.where(body_scores >= 0.5, 1, 0)
    scipy.misc.imsave(res_dir + str(idx) + '_edge_prob.png', edg_scores)
    scipy.misc.imsave(res_dir + str(idx) + '_edge_seg.png', edg_pred)
    scipy.misc.imsave(res_dir + str(idx) + '_body_prob.png', body_scores)
    scipy.misc.imsave(res_dir + str(idx) + '_body_seg.png', body_pred)

    feat_scores = np.squeeze(final_feat[idx])
    scipy.misc.imsave(res_dir + str(idx) + '_feat_visual.png', feat_scores)


dice_mean = np.mean(dice)
dice_std = np.std(dice)
hd95_mean = np.mean(hd95)
hd95_std = np.std(hd95)
acc_mean = np.mean(acc)
acc_std = np.std(acc)
sen_mean = np.mean(sen)
sen_std = np.std(sen)
spe_mean = np.mean(spe)
spe_std = np.std(spe)
pre_mean = np.mean(pre)
pre_std = np.std(pre)


print("\nDice: "+str(dice_mean)+" STD: "+str(dice_std))
print("\nSens: "+str(sen_mean)+" STD: "+str(sen_std))

para = np.vstack([dice, acc, sen, spe, pre, hd95, jac])

scio.savemat('prob_stats_BEG_SEGNET', {'para': para})


