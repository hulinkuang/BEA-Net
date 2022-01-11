from __future__ import division
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from networks import BEG_SegNet as M
import numpy as np
import scipy.io as scio


from medpy import metric




####################################  Load Data #####################################
te_data = np.load('test_img_normalized_demo.npy')
te_mask = np.load('test_mask_demo.npy')
te_mask = np.expand_dims(te_mask, axis=3)

model = M.unet(input_size=(256, 256, 3))
model.load_weights('trained_model.h5')

[predictions, edg] = model.predict(te_data, batch_size=8, verbose=1)



dice = np.zeros([8])
auc = np.zeros([8])
sen = np.zeros([8])
spe = np.zeros([8])
acc = np.zeros([8])
pre = np.zeros([8])
jac = np.zeros([8])
para = np.zeros([8, 7])
hd95 = np.zeros([8])
th = 0.5
for idx in range(8):

    print(idx)
    y_true = np.squeeze(te_mask[idx])
    y_true = np.where(y_true >= 0.5, 1, 0)
    y_scores = np.squeeze(predictions[idx])

    y_pred = np.where(y_scores >= th, 1, 0)

    # if np.max(y_pred) == 0:
    #     hd0 = 0
    # else:
    #     hd0 = metric.binary.hd95(y_pred, y_true)
    hd0 = metric.binary.hd95(y_pred, y_true)
    sens0 = metric.binary.sensitivity(y_pred, y_true)
    dice0 = metric.binary.dc(y_pred, y_true)
    spe0 = metric.binary.specificity(y_pred, y_true)
    pre0 = metric.binary.precision(y_pred, y_true)
    jc0 = metric.binary.jc(y_pred, y_true)
    # if jc0 < 0.65:
    #     jc0 = 0
    jac[idx] = jc0
    eid = np.where((y_pred - y_true) == 0)
    acc0 = len(y_pred[eid]) / (256 * 256)
    acc[idx] = acc0
    dice[idx] = dice0
    hd95[idx] = hd0
    spe[idx] = spe0
    sen[idx] = sens0
    pre[idx] = pre0


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
print("\nHD95: "+str(hd95_mean)+" STD: "+str(hd95_std))



