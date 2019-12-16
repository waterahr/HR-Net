import sys
sys.path.append("..")
import os
from keras.models import Model
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, concatenate, Lambda
from keras.utils import plot_model
#from spp.spp.SpatialPyramidPooling import SpatialPyramidPooling
import keras.backend as K
import numpy as np
from keras.models import Sequential
from keras.initializers import glorot_uniform


class hiarBayesResNet:
    @staticmethod
    def identity_block(X, f, filters, stage, block):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2, F3 = filters

        X_shortcut = X

        X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X
 
    @staticmethod
    def convolution_block(X, f, filters, stage, block, s=2):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        F1, F2, F3 = filters

        X_shortcut = X

        X = Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        X_shortcut = Conv2D(F3, (1,1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, name=bn_name_base + '1')(X_shortcut)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X
    

    @staticmethod
    def build(classes, input_shape = (224, 224, 3)):
        X_input = Input(input_shape)

        X = ZeroPadding2D((3, 3))(X_input)

        X = Conv2D(64, (7, 7), strides = (2,2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides = (2,2))(X)

        X = hiarBayesResNet.convolution_block(X, f = 3, filters = [64,64,256], stage = 2, block = 'a', s = 1)
        X = hiarBayesResNet.identity_block(X, 3, [64,64,256], stage=2, block='b')
        X = hiarBayesResNet.identity_block(X, 3, [64,64,256], stage=2, block='c')

        X = hiarBayesResNet.convolution_block(X, f = 3, filters = [128,128,512], stage = 3, block = 'a', s = 2)
        X = hiarBayesResNet.identity_block(X, 3, [128,128,512], stage=3, block='b')
        X = hiarBayesResNet.identity_block(X, 3, [128,128,512], stage=3, block='c')
        X = hiarBayesResNet.identity_block(X, 3, [128,128,512], stage=3, block='d')
        #fea_low = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv1_e')(X)
        #fea_low = hiarBayesResNet.convolution_block(X, f = 3, filters = [128,128,512], stage = 2, block = 'low', s = 1)
        fea_low = GlobalAveragePooling2D()(X)#fea_low
        #fea_low = Flatten()(X)
        fea_low = Dense(512, activation='relu')(fea_low)#fea_low
        #fea_low = Dense(512, activation='relu')(X)#fea_low
        #fea_low = GlobalAveragePooling2D()(fea_low)#fea_low

        X = hiarBayesResNet.convolution_block(X, f = 3, filters = [256,256,1024], stage = 4, block = 'a', s = 2)
        X = hiarBayesResNet.identity_block(X, 3, [256,256,1024], stage=4, block='b')
        X = hiarBayesResNet.identity_block(X, 3, [256,256,1024], stage=4, block='c')
        X = hiarBayesResNet.identity_block(X, 3, [256,256,1024], stage=4, block='d')    
        X = hiarBayesResNet.identity_block(X, 3, [256,256,1024], stage=4, block='e')
        X = hiarBayesResNet.identity_block(X, 3, [256,256,1024], stage=4, block='f')
        #fea_mid = Conv2D(2048, (3, 3), padding='same', activation='relu', name='conv2_e')(X)
        #fea_mid = hiarBayesResNet.convolution_block(X, f = 3, filters = [256,256,1024], stage = 4, block = 'mid', s = 1)
        fea_mid = GlobalAveragePooling2D()(X)#fea_mid
        #fea_mid = Flatten()(X)
        fea_mid = Dense(1024, activation='relu')(fea_mid)#fea_mid
        #fea_mid = Dense(1024, activation='relu')(X)
        #fea_mid = GlobalAveragePooling2D()(fea_mid)

        X = hiarBayesResNet.convolution_block(X, f = 3, filters = [512,512,2048], stage = 5, block = 'a', s = 2)
        X = hiarBayesResNet.identity_block(X, 3, [512,512,2048], stage=5, block='b')
        X = hiarBayesResNet.identity_block(X, 3, [512,512,2048], stage=5, block='c')
        #fea_hig = Conv2D(2048, (3, 3), padding='same', activation='relu', name='conv3_e')(X)
        #fea_hig = hiarBayesResNet.convolution_block(X, f = 3, filters = [512,512,2048], stage = 5, block = 'hig', s = 1)
        fea_hig = GlobalAveragePooling2D()(X)#fea_hig
        #fea_hig = Flatten()(X)
        fea_hig = Dense(1024, activation='relu')(fea_hig)#fea_hig
        #fea_hig = Dense(1024, activation='relu')(X)
        #fea_hig = GlobalAveragePooling2D()(fea_hig)

        #X = AveragePooling2D((2, 2), name='avg_pool')(X)

        #X = Flatten()(X)
        #X = Dense(classes[0], activation = 'softmax', name = 'fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
        predictions_low = Dense(classes[0], name="low", activation="sigmoid")(fea_low)#
        predictions_mid = Dense(classes[1], name="middle", activation="sigmoid")(fea_mid)#
        predictions_hig = Dense(classes[2], name="high", activation="sigmoid")(fea_hig)#
        predictions_priori = concatenate([predictions_low, predictions_mid], axis=1)
        predictions_hig_cond = Dense(classes[2], activation="sigmoid", name="high_cond")(predictions_priori)
        predictions_hig_posterior = Lambda(lambda x:x[1] * x[0], name="high_post")([predictions_hig_cond, predictions_hig])
        predictions = concatenate([predictions_low, predictions_mid, predictions_hig_posterior], axis=1)

        model = Model(inputs = X_input, outputs = predictions, name = 'ResNet50')
        model.load_weights("/home/anhaoran/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)
        return model

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    model = hiarBayesResNet.build([10, 20, 30])#因为googleNet默认输入32*32的图片
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file="../../results/models/hiarBayesGoogleLenet.png", show_shapes=True)