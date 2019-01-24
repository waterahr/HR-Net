from keras.layers.core import Layer
from keras.layers import Conv2D, Dense, Flatten
from keras.engine import InputSpec
from keras import initializers
from keras.utils import conv_utils
from keras.regularizers import l2
import six
from keras.utils import plot_model
from keras import backend as K
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations
import random
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
import numpy as np
import os
from angular_nonlinear_activations import *

class TanConv(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same", alpha=1, activation="square_cosine",
                 kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4), **kwargs):
        
        #self.nInputPlane = in_channels
        self.nOutputPlane = filters
        self.kernelSize = kernel_size
        self.alpha_conv = alpha
        self.strides = strides
        self.padding = padding
        self.kernelWeights = None
        self.rho = None
        self.alpha = alpha
        self.activation = activation
        self.kernelInitializer = kernel_initializer
        self.kernelRegularizer = kernel_regularizer
        #self.LBCNN.weight.requires_grad=False        
        super(TanConv, self).__init__(**kwargs)
        
    def build(self, input_shape):
        #print("****", input_shape)
        self.nInputPlane = input_shape[-1]
        #init weight
        
        initial_weights = K.random_normal((self.kernelSize[0], self.kernelSize[0], self.nInputPlane, self.nOutputPlane), 0, 1)
        self.kernelWeights = K.variable(initial_weights, name='{}_kernel_weights'.format(self.name))
        self.rho = K.variable(self.alpha, dtype='float32', name='{}_rho_weights'.format(self.name))
        #self.kernelWeights = K.ones((self.kernelSize[0], self.kernelSize[0], self.nInputPlane, self.nOutputPlane))
        """
        self.kernelWeights = self.add_weight(name='kernelWeights', 
                                      shape=(self.kernelSize[0], self.kernelSize[0], self.nInputPlane, self.nOutputPlane),
                                      initializer='uniform',
                                      trainable=True)"""
        
        self.trainable_weights = [self.kernelWeights, self.rho]
        #print(self.kernelWeights)
        #print(self.kernelWeights.shape)
        super(TanConv, self).build(input_shape)
        
    def call(self, inputs, mask=None):
        #print(inputs * inputs)
        one_kernel = K.ones((self.kernelSize[0], self.kernelSize[0], self.nInputPlane, self.nOutputPlane))
        inputs_norm =  K.conv2d(inputs * inputs, one_kernel, strides = self.strides, padding = self.padding)
        inputs_norm = K.sqrt(inputs_norm)
        #print(inputs_norm)#(?, 2, 2, 1)
        conv =  K.conv2d(inputs, self.kernelWeights, strides = self.strides, padding = self.padding)
        #print("+++", conv / ( inputs_norm * K.sqrt(K.sum(self.kernelWeights*self.kernelWeights))))#(?, 2, 2, 1)
        #print(K.sqrt(K.sum(self.kernelWeights*self.kernelWeights)))#()
        theta = conv / ( inputs_norm * K.sqrt(K.sum(self.kernelWeights*self.kernelWeights)))
        #"""
        if self.activation == "linear":
            g = linear_angular_activation(theta)
        elif self.activation == "cosine":
            g = cosine_angular_activation(theta)
        elif self.activation == "sigmoid":
            g = sigmoid_angular_activation(theta)
        elif self.activation == "square_cosine":
            g = square_cosine_angular_activation(theta)
        #"""
        #print(self.alpha_conv)
        h = self.alpha_conv * K.tanh(inputs_norm/self.rho)
        return h * g
        
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        h_axis, w_axis = 1, 2
        stride_h, stride_w = self.strides
        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernelSize
        
        out_height = conv_utils.conv_output_length(height, kernel_h, self.padding, stride_h)
        out_width = conv_utils.conv_output_length(width, kernel_w, self.padding, stride_w)
        output_shape = (batch_size, out_height, out_width, self.nOutputPlane)
        #print(output_shape)
        return output_shape

    def get_config(self):
        config = {#"in_channels" : self.nInputPlane,    
                  "out_channels" : self.nOutputPlane, 
                  "kernel_size" : self.kernelSize, 
                  "strides" : self.strides,
                  "alpha" : self.alpha_conv,
                  "rho" : self.rho,
                  "padding" : self.padding, 
                  "activation": self.activation,
                  "kernel_initializer" : self.kernelInitializer, 
                  "kernel_regularizer" : self.kernelRegularizer}
        base_config = super(TanConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    input_t = Input(shape=(3,3,1))
    conv_t = TanConv(filters=1, kernel_size=(2, 2), strides=(1, 1), padding="valid")(input_t)
    #print(TanConv(filters=1, kernel_size=(2, 2), strides=(1, 1), padding="valid").get_config())
    conv_t = Flatten()(conv_t)
    pred = Dense(2, activation="sigmoid")(conv_t)
    model1_t = Model(input_t, pred)
    #"""
    model1_t.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model1_t.summary()
    
    train_x = []
    train_y = []
    for i in range(100):
        tmp = []
        for j in range(9):
            tmp.append(random.gauss(1, 2))
        tmp = np.array(tmp).reshape((3, 3, 1))
        train_x.append(tmp)
        train_y.append([0, 1])
    for i in range(100):
        tmp = []
        for j in range(9):
            tmp.append(random.gauss(0, 1))
        tmp = np.array(tmp).reshape((3, 3, 1))
        train_x.append(tmp)
        train_y.append([1, 0])
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print(train_x.shape)
    print(train_y.shape)
    
    model1_t.fit(train_x, train_y, epochs=10)
    #"""
    
    a = np.asarray([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]).reshape(1, 3, 3, 1)
    print(a.shape)
    res1_t = model1_t.predict(a)
    print(res1_t)