"""
There is a problem :
the angular feature must be sparse_categorical_crossentropy;
the radial feature should be used as a regression situation?
the radial feature must belong to same angular.
=>1\one solution(OEDCGoogLeNetSPP_lowerBody) is augular for coarse classification
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
from keras.layers import Embedding, Lambda
import keras.backend as K
import numpy as np
from keras.models import Sequential
import os
import random
#from angular_losses import GenerativeAngularSoftmaxLoss



class OEDCGoogLeNetSPP_lowerBody:
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
        x = TanConv(filters=nb_filter, kernel_size=kernel_size, strides=strides, padding=padding, activation="square_cosine", name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x
 
    @staticmethod
    def Inception(x, nb_filter):
        branch1x1 = OEDCGoogLeNetSPP_lowerBody.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
     
        branch3x3 = OEDCGoogLeNetSPP_lowerBody.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
        branch3x3 = OEDCGoogLeNetSPP_lowerBody.Conv2d_BN(branch3x3, nb_filter,(3,3), padding='same', strides=(1,1), name=None)
     
        branch5x5 = OEDCGoogLeNetSPP_lowerBody.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1),name=None)
        branch5x5 = OEDCGoogLeNetSPP_lowerBody.Conv2d_BN(branch5x5, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
        branchpool = OEDCGoogLeNetSPP_lowerBody.Conv2d_BN(branchpool, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
     
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
    def build(width, height, depth, classes_coarse, classes_fine, classes_radial, pooling_regions = [1, 3]):
        """Create the sharing model"""
        #'classes_fine' as a list, and the length must be equal to 'classes_coarse'
        inpt = Input(shape=(width, height, depth))
        #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
        x = OEDCGoogLeNetSPP_lowerBody.Conv2d_BN(inpt, 64, (7,7), strides=(2,2), padding='same')
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        x = OEDCGoogLeNetSPP_lowerBody.Conv2d_BN(x, 192, (3,3), strides=(1,1), padding='same')
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        x = OEDCGoogLeNetSPP_lowerBody.Inception(x, 64)#256
        x = OEDCGoogLeNetSPP_lowerBody.Inception(x, 120)#480
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        x = OEDCGoogLeNetSPP_lowerBody.Inception(x, 128)#512
        spp_low = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv1_e')(x)
        #spp_low = TanConv(filters=512, kernel_size=(3, 3), padding='same', activation="square_cosine", name='conv1_e')(x)
        spp_low = SpatialPyramidPooling(pooling_regions)(spp_low)
        spp_low = Dense(512, activation='relu')(spp_low)
        x = OEDCGoogLeNetSPP_lowerBody.Inception(x, 128)
        x = OEDCGoogLeNetSPP_lowerBody.Inception(x, 128)
        x = OEDCGoogLeNetSPP_lowerBody.Inception(x, 132)#528
        spp_mid = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv2_e')(x)
        #spp_mid = TanConv(filters=512, kernel_size=(3, 3), padding='same', activation="square_cosine", name='conv2_e')(x)
        spp_mid = SpatialPyramidPooling(pooling_regions)(spp_mid)
        spp_mid = Dense(512, activation='relu')(spp_mid)	
        x = OEDCGoogLeNetSPP_lowerBody.Inception(x, 208)#832
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        x = OEDCGoogLeNetSPP_lowerBody.Inception(x, 208)
        x = OEDCGoogLeNetSPP_lowerBody.Inception(x, 256)#1024
        spp_hig = Conv2D(1024, (3, 3), padding='same', activation='relu', name='conv3_e')(x)
        #spp_hig = TanConv(filters=1024, kernel_size=(3, 3), padding='same', activation="square_cosine", name='conv3_e')(x)
        spp_hig = SpatialPyramidPooling([1, 2])(spp_hig)
        spp_hig = Dense(1024, activation='relu')(spp_hig)
        x = concatenate([spp_low, spp_mid, spp_hig], axis=1)#2048
        #x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)
        x = Dropout(0.4)(x)
        x = Dense(2048, activation='relu')(x)#decoupled feature vector
        
        """Create the predict model"""
        ###coarse decoupled the feature into two classes
        #x = BatchNormalization(axis=1, name='bn_angular_coarse')(x)
        #x = Dense(2048, activation='relu', name="dense_angular_coarse")(x)
        pred_coarse = Dense(classes_coarse, activation='softmax')(x)
        ###fine decoupled the feature into multi classes
        x_angular = BatchNormalization(axis=1, name='bn_angular_fine')(x)
        x_angular = Dense(1024, activation='relu', name="dense_angular_fine")(x_angular)
        pred_fine = Dense(classes_fine, activation='softmax')(x_angular)
        ###using the radial feature to deal with the style
        x_radial = Lambda(OEDCGoogLeNetSPP_lowerBody.l2_norm)(x)
        #x_radial = BatchNormalization(axis=1, name='bn_raidal_fine')(x)
        #x_radial = Dense(1024, activation='relu', name="dense_radial")(x_radial)
        pred_radial = Dense(classes_radial, activation='softmax')(x_radial)
        predictions = concatenate([pred_coarse, pred_fine, pred_radial], axis = 1, name="softmax_labels")
        #x = Dense(classes_coarse+classes_fine+classes_radial, activation="softmax")(x)
        
        """Create the train model"""
        #GA-Softmax Loss(Target) Layer
        #GenerativeAngularSoftmaxLoss()([x, ])
        
        """create the model"""
        #model_train = Model(inpt, )
        model_pred = Model(inpt, predictions, name='inception')
        # return the constructed network architecture
        return model_pred#model_train, model_pred

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    model = OEDCGoogLeNetSPP_lowerBody.build(None, None, 3, 2, 7, 5)#因为googleNet默认输入32*32的图片
    plot_model(model, to_file="../../results/OE-DC-GoogleLenet-SPP-lower.png", show_shapes=True)
