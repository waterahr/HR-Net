"""
There is a problem :
the angular feature must be sparse_categorical_crossentropy;
the radial feature should be used as a regression situation?
the radial feature must belong to same angular.
=>1\one solution(OEDCGoogLeNetSPP) is augular for coarse classification
  and radial is for fine classification(not suitable);
  2\the other solution(next think) may be angualr for fine and non-intersection
  classification and radial for location.
"""
import sys
sys.path.append("..")
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import plot_model
from spp.spp.SpatialPyramidPooling import SpatialPyramidPooling
from tanconv import TanConv
from keras.layers import Lambda
import keras.backend as K
import numpy as np
from keras.models import Sequential
import os
import random



class OEDCGoogLeNetSPP:
    @staticmethod
    def l2_norm(x):
        #print(K.int_shape(x))
        #print(K.int_shape(K.sum(x**2, axis=1)))
        #print(K.int_shape(K.sqrt(K.sum(x**2, axis=1))))
        return K.reshape(K.sqrt(K.sum(x**2, axis=1)), shape=(-1, 1))
    
    @staticmethod
    def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1,1), name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
     
        #x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = TanConv(filters=nb_filter, kernel_size=kernel_size, strides=strides, padding=padding, name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x
 
    @staticmethod
    def Inception(x, nb_filter):
        branch1x1 = OEDCGoogLeNetSPP.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
     
        branch3x3 = OEDCGoogLeNetSPP.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
        branch3x3 = OEDCGoogLeNetSPP.Conv2d_BN(branch3x3, nb_filter,(3,3), padding='same', strides=(1,1), name=None)
     
        branch5x5 = OEDCGoogLeNetSPP.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1),name=None)
        branch5x5 = OEDCGoogLeNetSPP.Conv2d_BN(branch5x5, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
        branchpool = OEDCGoogLeNetSPP.Conv2d_BN(branchpool, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
     
        x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)
     
        return x

    """
    @staticmethod
    def SPP(x, pooling_regions):
        dim_ordering = K.image_dim_ordering()
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        if dim_ordering == 'th':
            input_shape = (num_channels, None, None)
        elif dim_ordering == 'tf':
        input_shape = (None, None, num_channels)
        model = Sequential()
        model.add(SpatialPyramidPooling(pooling_regions, input_shape=input_shape))
        
        return model.predict(x)
    """
    

    @staticmethod
    def build(width, height, depth, classes_coarse, classes_fine, pooling_regions = [1, 3]):
        #'classes_fine' as a list, and the length must be equal to 'classes_coarse'
        assert(type(classes_fine) is list, 'The class_fine-grained must be a list, containing the coarse classes\' fine class number')
        assert(len(classes_fine) == classes_coarse, 'The length of the list(class_fine-grained) must be equal to classes_coarse('+str(classes_coarse)+')')
        inpt = Input(shape=(width, height, depth))
        #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
        x = OEDCGoogLeNetSPP.Conv2d_BN(inpt, 64, (7,7), strides=(2,2), padding='same')
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        x = OEDCGoogLeNetSPP.Conv2d_BN(x, 192, (3,3), strides=(1,1), padding='same')
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        x = OEDCGoogLeNetSPP.Inception(x, 64)#256
        x = OEDCGoogLeNetSPP.Inception(x, 120)#480
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        x = OEDCGoogLeNetSPP.Inception(x, 128)#512
        spp_low = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv1_e')(x)
        spp_low = SpatialPyramidPooling(pooling_regions)(spp_low)
        spp_low = Dense(512, activation='relu')(spp_low)
        x = OEDCGoogLeNetSPP.Inception(x, 128)
        x = OEDCGoogLeNetSPP.Inception(x, 128)
        x = OEDCGoogLeNetSPP.Inception(x, 132)#528
        spp_mid = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv2_e')(x)
        spp_mid = SpatialPyramidPooling(pooling_regions)(spp_mid)
        spp_mid = Dense(512, activation='relu')(spp_mid)	
        x = OEDCGoogLeNetSPP.Inception(x, 208)#832
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        x = OEDCGoogLeNetSPP.Inception(x, 208)
        x = OEDCGoogLeNetSPP.Inception(x, 256)#1024
        spp_hig = Conv2D(1024, (3, 3), padding='same', activation='relu', name='conv3_e')(x)
        spp_hig = SpatialPyramidPooling([1, 2])(spp_hig)
        spp_hig = Dense(1024, activation='relu')(spp_hig)
        x = concatenate([spp_low, spp_mid, spp_hig], axis=1)#2048
        #x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)
        x = Dropout(0.4)(x)
        x = Dense(2048, activation='relu')(x)#decoupled feature vector
        x_angular = BatchNormalization(axis=1, name='bn_angular')(x)
        x_angular = Dense(1024, activation='relu')(x_angular)
        x_angular = Dense(classes_coarse, activation='softmax')(x_angular)
        x_radial = Lambda(OEDCGoogLeNetSPP.l2_norm)(x)
        x_radial = Dense(1024, activation='relu')(x_radial)
        x_radial = [Dense(class_idx, activation='softmax')(x_radial) for class_idx in classes_fine]
        x = concatenate(x_radial, axis=1)
        x = concatenate([x_angular, x], axis = 1)
        # create the model
        model = Model(inpt, x, name='inception')
        # return the constructed network architecture
        return model

if __name__ == "__main__":
    model = OEDCGoogLeNetSPP.build(None, None, 3, 7, [3, 7, 11, 6, 7, 12, 15])#因为googleNet默认输入32*32的图片
    plot_model(model, to_file="../../results/OE-DC-GoogleLenet-SPP.png", show_shapes=True)
