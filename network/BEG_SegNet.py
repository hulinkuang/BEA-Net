

import numpy as np
import os

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.backend import tf as ktf
from keras import backend as K
import tensorflow as tf
from keras.layers import GaussianNoise

from keras.engine.topology import Layer


class MyLayer(Layer):

    def __init__(self, output_c, **kwargs):
        self.output_c = output_c
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # # 为该层创建一个可训练的权重
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        in_shape = x.get_shape()
        image = K.tf.image.sobel_edges(x)
        image = image ** 2
        seg_edg0 = K.tf.reduce_sum(image, axis=-1)
        # seg_edg = K.sqrt(K.tf.reduce_sum(image, axis=-1))
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        image = K.tf.image.sobel_edges(x1)
        image = image ** 2
        seg_edg1 = K.tf.reduce_sum(image, axis=-1)
        seg_edg1 = UpSampling2D(size=(2, 2))(seg_edg1)

        x2 = MaxPooling2D(pool_size=(4, 4), padding='same')(x)
        image = K.tf.image.sobel_edges(x2)
        image = image ** 2
        seg_edg2 = K.tf.reduce_sum(image, axis=-1)
        seg_edg2 = UpSampling2D(size=(4, 4))(seg_edg2)
        seg_edg = K.concatenate([seg_edg0, seg_edg1, seg_edg2], axis=3)
        seg_edg = K.tf.layers.Conv2D(self.output_c, 1, padding='same')(seg_edg)
        seg_edg = K.tf.contrib.layers.batch_norm(seg_edg)  # BatchNormalization(axis=3)(image)
        seg_edg = K.tf.nn.relu(seg_edg)
        # x1 = K.tf.layers.MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(x)
        # image = K.tf.image.sobel_edges(x1)
        # image = image ** 2
        # seg_edg1 = K.tf.reduce_sum(image, axis=-1)
        # seg_edg1 = K.tf.keras.layers.UpSampling2D(size=(2, 2))(seg_edg1)
        #
        # x2 = K.tf.layers.MaxPooling2D(pool_size=(4, 4), strides=1, padding='same')(x)
        # image = K.tf.image.sobel_edges(x2)
        # image = image ** 2
        # seg_edg2 = K.tf.reduce_sum(image, axis=-1)
        # seg_edg2 = K.tf.keras.layers.UpSampling2D(size=(4, 4))(seg_edg2)
        # seg_edg = concatenate([seg_edg0, seg_edg1, seg_edg2], axis=3)
        # seg_edg = K.tf.layers.Conv2D(128, 1, padding='same')(seg_edg)
        # seg_edg = K.tf.contrib.layers.batch_norm(seg_edg)  # BatchNormalization(axis=3)(image)
        # seg_edg = K.tf.nn.relu(seg_edg)

        return seg_edg
        # return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape


def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    # smooth = 1.
    # y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    # intersection = y_true_f * y_pred_f
    # score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    score = dice_coef(y_true, y_pred)
    return 1. - score


def joint_loss(y_true, y_pred):
    w1 = 0.8
    w2 = 0.2
    lossseg = w1 * dice_loss(y_true, y_pred) + w2 * cross_entropy_balanced(y_true, y_pred)

    return lossseg


def generalized_dice_loss(y_true, y_pred):
    # Compute weights: "the contribution of each label is corrected by the inverse of its volume"
    w = K.sum(y_true, axis=(0, 1, 2, 3))
    w = 1 / (w ** 2 + 0.00001)
    # w为各个类别的权重，占比越大，权重越小
    # Compute gen dice coef:
    numerator = y_true * y_pred
    numerator = w * K.sum(numerator, axis=(0, 1, 2, 3))
    numerator = K.sum(numerator)

    denominator = y_true + y_pred
    denominator = w * K.sum(denominator, axis=(0, 1, 2, 3))
    denominator = K.sum(denominator)

    gen_dice_coef = numerator / denominator

    return 1 - 2 * gen_dice_coef


def Interp(x, shape):
    new_height, new_width = shape
    resized = ktf.image.resize_images(x, [new_height, new_width],
                                      align_corners=True)
    return resized


def side_brach(x, factor):
    x = Conv2D(1, 1, activation=None, padding='same')(x)
    kernel_size = (2 * factor, 2 * factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)
    return x


def SqueezeBodyEdge(x, filter, H, W):
    down = Conv2D(filters=filter, kernel_size=3, strides=2, kernel_initializer='he_normal')(x)
    down = BatchNormalization(axis=3)(down)
    down = Activation('relu')(down)

    down = Conv2D(filters=filter, kernel_size=3, strides=2, kernel_initializer='he_normal')(down)
    down = BatchNormalization(axis=3)(down)
    down = Activation('relu')(down)

    seg_down = Lambda(Interp, arguments={'shape': (np.int32(H), np.int32(W))})(down)
    flow = concatenate([x, seg_down], axis=3)
    flow = Conv2D(filters=filter, kernel_size=1, kernel_initializer='he_normal')(flow)
    flow = BatchNormalization(axis=3)(flow)
    seg_flow = Activation('relu')(flow)

    seg_edg = Subtract()([x, seg_flow])
    return seg_flow  # , seg_edg


def msstc256(x):
    x1 = Conv2D(128, 1, activation=None, padding='same')(x)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)
    x2 = Conv2D(64, 3, activation=None, padding='same')(x1)
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)
    x3 = Conv2D(64, 5, activation=None, padding='same')(x2)
    x3 = BatchNormalization(axis=3)(x3)
    x3 = Activation('relu')(x3)
    x = concatenate([x1, x2, x3], axis=3)
    x = Conv2D(256, 1, activation=None, padding='same')(x)
    return x


def msstc512(x):
    x1 = Conv2D(256, 1, activation=None, padding='same')(x)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)
    x2 = Conv2D(128, 3, activation=None, padding='same')(x1)
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)
    x3 = Conv2D(128, 5, activation=None, padding='same')(x2)
    x3 = BatchNormalization(axis=3)(x3)
    x3 = Activation('relu')(x3)
    x = concatenate([x1, x2, x3], axis=3)
    x = Conv2D(512, 1, activation=None, padding='same')(x)
    return x


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    N1 = input_size[0]
    N2 = input_size[1]
    # encode
    input = GaussianNoise(0.005)(inputs)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(input)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)

    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1_1 = Conv2D(64, 1, padding='same', kernel_initializer='he_normal')(conv1)
    pool1_1 = BatchNormalization(axis=3)(pool1_1)
    pool1_1 = Activation('relu')(pool1_1)
    pool1_1 = AveragePooling2D(pool_size=(2, 2))(pool1_1)
    pool1 = concatenate([pool1, pool1_1], axis=3)

    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2_1 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(conv2)
    pool2_1 = BatchNormalization(axis=3)(pool2_1)
    pool2_1 = Activation('relu')(pool2_1)
    pool2_1 = AveragePooling2D(pool_size=(2, 2))(pool2_1)
    pool2 = concatenate([pool2, pool2_1], axis=3)

    conv3 = msstc256(pool2)
    # conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)

    # conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = msstc256(conv3)
    conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3_1 = Conv2D(256, 1, padding='same', kernel_initializer='he_normal')(conv3)
    pool3_1 = BatchNormalization(axis=3)(pool3_1)
    pool3_1 = Activation('relu')(pool3_1)
    pool3_1 = AveragePooling2D(pool_size=(2, 2))(pool3_1)
    pool3 = concatenate([pool3, pool3_1], axis=3)

    # conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = msstc512(pool3)
    conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Activation('relu')(conv4)

    # conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = msstc512(conv4)
    conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.5)(conv4)

    # # psp
    # psp_conv1 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(conv4)
    # psp_conv1 = BatchNormalization(axis=3)(psp_conv1)
    # psp_conv1 = AveragePooling2D(pool_size=(np.int32(N1 / 8), np.int32(N2 / 8)))(psp_conv1)
    # psp_conv1 = Activation('relu')(psp_conv1)
    # psp_conv1 = Lambda(Interp, arguments={'shape': (np.int32(N1 / 8), np.int32(N2 / 8))})(psp_conv1)
    #
    # psp_conv2 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(conv4)
    # psp_conv2 = BatchNormalization(axis=3)(psp_conv2)
    # psp_conv2 = Activation('relu')(psp_conv2)
    # psp_conv2 = AveragePooling2D(pool_size=(4, 4))(psp_conv2)
    # psp_conv2 = Lambda(Interp, arguments={'shape': (np.int32(N1 / 8), np.int32(N2 / 8))})(psp_conv2)
    #
    # psp_conv3 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(conv4)
    # psp_conv3 = BatchNormalization(axis=3)(psp_conv3)
    # psp_conv3 = Activation('relu')(psp_conv3)
    # psp_conv3 = AveragePooling2D(pool_size=(8, 8))(psp_conv3)
    # psp_conv3 = Lambda(Interp, arguments={'shape': (np.int32(N1 / 8), np.int32(N2 / 8))})(psp_conv3)
    #
    # psp_conv4 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(conv4)
    # psp_conv4 = BatchNormalization(axis=3)(psp_conv4)
    # psp_conv4 = Activation('relu')(psp_conv4)
    # psp_conv4 = AveragePooling2D(pool_size=(16, 16))(psp_conv4)
    # psp_conv4 = Lambda(Interp, arguments={'shape': (np.int32(N1 / 8), np.int32(N2 / 8))})(psp_conv4)

    # # layer5 = concatenate([conv4, psp_conv1, psp_conv2, psp_conv3, psp_conv4], axis=3)
    # layer5 = msstc512(conv4)
    # layer5 = Conv2D(512, 1, padding='same', kernel_initializer='he_normal')(layer5)
    # layer5 = BatchNormalization(axis=3)(layer5)
    # layer5 = Activation('relu')(layer5)
    #
    # # layer5 = msstc512(layer5)
    # layer5 = Conv2D(512, 1, padding='same', kernel_initializer='he_normal')(layer5)
    # layer5 = BatchNormalization(axis=3)(layer5)
    # layer5 = Activation('relu')(layer5)
    #
    # layer5 = Dropout(0.5)(layer5)

    # body, d_edge = SqueezeBodyEdge(layer5, 512, np.int32(N1 / 8), np.int32(N2 / 8))
    body = SqueezeBodyEdge(conv4, 512, np.int32(N1 / 8), np.int32(N2 / 8))
    d_edge = MyLayer(512)(conv4)
    # body = Subtract()([conv4, d_edge])
    # decode
    # upsample1
    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(body))
    up6 = Lambda(Interp, arguments={'shape': (np.int32(N1 / 4), np.int32(N2 / 4))})(up6)
    merge6 = concatenate([conv3, up6], axis=3)

    # conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = msstc256(merge6)

    conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    # conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = msstc256(conv6)
    conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    # edge1
    d_edge = Lambda(Interp, arguments={'shape': (np.int32(N1 / 4), np.int32(N2 / 4))})(d_edge)
    # d_edge = Conv2D(256, 1, kernel_initializer='he_normal')(d_edge)
    d_edge = msstc256(d_edge)
    d_edge = BatchNormalization(axis=3)(d_edge)
    d_edge = Activation('relu')(d_edge)

    # body1, edge1 = Lambda(LPBodyEdge)(conv6)#LPBodyEdge(conv6)#Lambda(LPBodyEdge, arguments={'C': 256})(conv6)#SqueezeBodyEdge(conv6, 256, np.int32(N1 / 4), np.int32(N2 / 4))
    body1 = SqueezeBodyEdge(conv6, 256, np.int32(N1 / 4), np.int32(N2 / 4))
    edge1 = MyLayer(256)(conv6)
    # body1 = Subtract()([conv6, edge1])
    edge1 = concatenate([edge1, d_edge], axis=3)

    # upsample2
    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(body1))
    up7 = Lambda(Interp, arguments={'shape': (np.int32(N1 / 2), np.int32(N2 / 2))})(up7)
    merge7 = concatenate([conv2, up7], axis=3)

    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    # edge2
    edge1 = Lambda(Interp, arguments={'shape': (np.int32(N1 / 2), np.int32(N2 / 2))})(edge1)
    edge1 = Conv2D(128, 1, kernel_initializer='he_normal')(edge1)
    edge1 = BatchNormalization(axis=3)(edge1)
    edge1 = Activation('relu')(edge1)

    # body2, edge2 = Lambda(LPBodyEdge)(conv7)#LPBodyEdge(conv7)#Lambda(LPBodyEdge)(conv7)#Lambda(LPBodyEdge, arguments={'C': 128})(conv7)#SqueezeBodyEdge(conv7, 128, np.int32(N1 / 2), np.int32(N2 / 2))
    body2 = SqueezeBodyEdge(conv7, 128, np.int32(N1 / 2), np.int32(N2 / 2))
    edge2 = MyLayer(128)(conv7)
    # body2 = Subtract()([conv7, edge2])
    edge2 = concatenate([edge2, edge1], axis=3)

    # upsample
    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(body2))
    up8 = Lambda(Interp, arguments={'shape': (np.int32(N1), np.int32(N2))})(up8)
    merge8 = concatenate([conv1, up8], axis=3)

    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    # edge3
    edge2 = Lambda(Interp, arguments={'shape': (np.int32(N1), np.int32(N2))})(edge2)
    edge2 = Conv2D(64, 1, kernel_initializer='he_normal')(edge2)
    edge2 = BatchNormalization(axis=3)(edge2)
    edge2 = Activation('relu')(edge2)

    # body3, edge3 = Lambda(LPBodyEdge)(conv8)#LPBodyEdge(conv8)#Lambda(LPBodyEdge)(conv8)#Lambda(LPBodyEdge, arguments={'C': 64})(conv8)#SqueezeBodyEdge(conv8, 64, np.int32(N1 ), np.int32(N2 ))
    body3 = SqueezeBodyEdge(conv8, 64, np.int32(N1), np.int32(N2))
    edge3 = MyLayer(64)(conv8)
    # body3 = Subtract()([conv8, edge3])
    edge3 = concatenate([edge3, edge2], axis=3)

    conv8_1 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(body3)
    conv9 = Conv2D(1, 1, activation='sigmoid')(conv8_1)

    # edg
    edg = Conv2D(1, 1)(edge3)
    edg = Activation('sigmoid', name='edg')(edg)

    final_seg = concatenate([conv9, edg], axis=3)
    final_seg = Conv2D(1, 1, activation='sigmoid', name='final_seg')(final_seg)

    model = Model(input=inputs, output=[final_seg, edg])

    # model.compile(optimizer=Adam(lr=1e-4), loss={'final_seg': joint_loss, 'edg': cross_entropy_balanced},
    #               loss_weights={'final_seg': 1, 'edg': 0.1},
    #               metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=1e-4), loss={'final_seg': joint_loss, 'edg': 'binary_crossentropy'},
                  loss_weights={'final_seg': 1.5, 'edg': 0.1},
                  metrics=['accuracy'])
    # cross_entropy_balanced binary_crossentropy  ['accuracy', dice_coef] ['accuracy']
    # {'final_seg': dice_coef, 'edg': 'accuracy'}
    # model.compile(optimizer=Adam(lr=1e-4), loss={'final_seg': 'binary_crossentropy', 'edg': 'binary_crossentropy'},
    #               loss_weights={'final_seg': 1, 'edg': 0.1},
    #               metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
