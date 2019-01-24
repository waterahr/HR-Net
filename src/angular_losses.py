from keras.layers.core import Layer
from keras import backend as K
import json
import numpy as np
import math
from angular_nonlinear_activations import *

def bayes_binary_crossentropy_in(y_true, y_pred, alpha, label):
    #alpha is a N*N matrix, whose i'th row representing the i'th attribute rely on
    #alpha.shape[0] == alpha.shape[1] == (K.int_shape(y_pred)[1]
    #return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    """
    #####loss
    loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    #print(K.shape(loss))
    for i in range(K.int_shape(y_pred)[1]):
        n = np.sum(alpha[i])
        if n == 0:
            continue
        #print(label * alpha[i])
        s = np.sum(label * alpha[i], axis=1)
        #loss -= (1.0 / n * s * K.log(y_pred[:, i]) )#+ (1 - 1.0 / n * s) * K.log(1 - y_pred[:, i]))
    """
    #"""
    #####loss_ifelse
    loss = K.zeros_like(K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1))
    #print(K.shape(loss))
    for i in range(K.int_shape(y_pred)[1]):
        n = np.sum(alpha[i])
        #print(label * alpha[i])
        if n != 0:
            s = np.sum(label * alpha[i], axis=1)
            loss -= (0.2 / n * s * K.log(y_pred[:, i]) + 0.8 * y_true[:, i] * K.log(y_pred[:, i]) + (1 - y_true[:, i]) * K.log(1 - y_pred[:, i]))
        else:
            loss -= (y_true[:, i] * K.log(y_pred[:, i]) + (1 - y_true[:, i]) * K.log(1 - y_pred[:, i]))
    #"""
    """
    #####loss2
    loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    #print(K.shape(loss))
    for i in range(K.int_shape(y_pred)[1]):
        n = np.sum(alpha[i])
        if n == 0:
            continue
        #print(label * alpha[i])
        s = K.sum(y_pred * alpha[i], axis=1)
        loss -= (1.0 / n * s * K.log(y_pred[:, i]) )#+ (1 - 1.0 / n * s) * K.log(1 - y_pred[:, i]))
    """
    """
    #####loss2_ifelse
    loss = K.zeros_like(K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1))
    #print(K.shape(loss))
    for i in range(K.int_shape(y_pred)[1]):
        n = np.sum(alpha[i])
        #print(label * alpha[i])
        if n != 0:
            s = K.sum(y_pred * alpha[i], axis=1)
            loss -= (0.2 / n * s * K.log(y_pred[:, i]) + 0.8 * y_true[:, i] * K.log(y_pred[:, i]) + (1 - y_true[:, i]) * K.log(1 - y_pred[:, i]))#y_true[:, i]
        else:
            loss -= (y_true[:, i] * K.log(y_pred[:, i]) + (1 - y_true[:, i]) * K.log(1 - y_pred[:, i]))
    """
    return loss#K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

def bayes_binary_crossentropy(alpha, label):
    def loss_interface(y_true, y_pred):
        return bayes_binary_crossentropy_in(y_true, y_pred, alpha, label)
    return loss_interface

def weighted_binary_crossentropy(alpha):

    def loss_interface(y_true, y_pred):
        """
        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_pred, y_true)

        # Apply the weights
        weight_vector = y_true * np.exp(-alpha) + (1. - y_true) * 1
        weighted_b_ce = weight_vector * b_ce
        """
        
        print(y_pred)
        print(1.0-y_pred)
        print(y_true)
        print(1.0-y_true)
        print(K.log(y_pred))
        """
        Tensor("dense_2/Sigmoid:0", shape=(?, 51), dtype=float32)
        Tensor("loss/dense_2_loss/sub:0", shape=(?, 51), dtype=float32)
        Tensor("dense_2_target:0", shape=(?, ?), dtype=float32)
        Tensor("loss/dense_2_loss/sub_1:0", shape=(?, ?), dtype=float32)
        Tensor("loss/dense_2_loss/Log:0", shape=(?, 51), dtype=float32)
        """
        #b_ce = K.sum(-y_pred * K.log(y_true) - (1.0 - y_pred) * K.log(1.0 - y_true), axis=-1)
        ###NaN
        logits = K.log(y_pred) - K.log(1.0 - y_pred)
        b_ce = logits - logits * y_true - K.log(y_pred)
        print(b_ce)#Tensor("loss/dense_2_loss/Sum:0", shape=(?,), dtype=float32)
        weighted_b_ce = logits - logits * y_true - (y_true * alpha + 1 - y_true) * K.log(y_pred)


        # Return the mean error
        #return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
        return K.mean(weighted_b_ce, axis=-1)

    return loss_interface


def weighted_categorical_crossentropy_in(y_true, y_pred, alpha):
    #print(data)
    #print(K.int_shape(y_pred))(None, 65)
    loss = K.zeros_like(K.categorical_crossentropy(y_true, y_pred))
    #print(loss)#Tensor("loss/concatenate_10_loss/zeros_like:0", shape=(?,), dtype=float32)
    #print(K.int_shape(loss))#(None,)
    #print(K.log(y_pred[:, 0]))#Tensor("loss/concatenate_10_loss/Log_1:0", shape=(?,), dtype=float32)
    #print(y_true[:, 0] * K.log(y_pred[:, 0]))#Tensor("loss/concatenate_10_loss/mul_1:0", shape=(?,), dtype=float32)
    for i in range(K.int_shape(y_pred)[1]):
        loss += 0.5 * (y_true[:, i] * K.log(y_pred[:, i]) / alpha[i] + (1 - y_true[:, i]) * K.log(1 - y_pred[:, i]) / (1 - alpha[i]))
        #loss += y_true[:, i] * K.log(y_pred[:, i]) + (1 - y_true[:, i]) * K.log(1 - y_pred[:, i])
    return loss

def weighted_categorical_crossentropy(alpha):
    def loss_interface(y_true, y_pred):
        return weighted_categorical_crossentropy_in(y_true, y_pred, alpha)
    return loss_interface

def coarse_to_fine_categorical_crossentropy_lowerbody_in(y_true, y_pred, alpha):
    #print(alpha)
    #print(K.int_shape(y_true))#(None, None)
    #print(K.int_shape(y_pred))#(None, 14)
    """
    loss = K.categorical_crossentropy(y_true[:, :2], y_pred[:, :2]) * 0.5
    loss = loss + K.categorical_crossentropy(y_true[:, 2:7], y_pred[:, 2:7]) * 0.2
    loss = loss + K.categorical_crossentropy(y_true[:, 7:9], y_pred[:, 7:9]) * 0.2
    loss = loss + K.categorical_crossentropy(y_true[:, 9:14], y_pred[:, 9:14]) * 0.1
    """
    loss = K.zeros_like(K.categorical_crossentropy(y_true, y_pred))
    for i in range(K.int_shape(y_pred)[1]):
        loss += 0.5 * (y_true[:, i] * K.log(y_pred[:, i]) / alpha[i] + y_pred[:, i] * K.log(y_true[:, i]) / (1 - alpha[i]))

    return loss#K.sparse_categorical_crossentropy(y_true, y_pred)

def coarse_to_fine_categorical_crossentropy_lowerbody(alpha):
    def loss_interface(y_true, y_pred):
        return coarse_to_fine_categorical_crossentropy_lowerbody_in(y_true, y_pred, alpha)
    return loss_interface


"""
class GenerativeAngularSoftmaxLoss(Layer):
    def __init__(self, margin=1, activation="square_cosine", **kwargs):
        self.margin = margin
        self.activation = activation
        super(GenerativeAngularSoftmaxLoss, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(GenerativeAngularSoftmaxLoss, self).build(input_shape)

    def call(self, inputs):
        x = inputs[0]
        x = K.sqrt(K.sum(K.pow(x, 2), axis=1))
        theta = inputs[1]
        if self.activation == "linear":
            g = linear_angular_activation(theta)
        elif self.activation == "cosine":
            g = cosine_angular_activation(theta)
        elif self.activation == "sigmoid":
            g = sigmoid_angular_activation(theta)
        elif self.activation == "square_cosine":
            g = square_cosine_angular_activation(theta)
        return x
    
    def get_config(self):
        config = {"margin": self.margin,
                 "activation": self.activation}
        base_config = super(GenerativeAngularSoftmaxLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
"""