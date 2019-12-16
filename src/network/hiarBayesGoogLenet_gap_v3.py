import sys
sys.path.append("..")
import os
from keras.models import Model
from keras.layers import Activation, Input, Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, GlobalAveragePooling2D, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
#from spp.spp.SpatialPyramidPooling import SpatialPyramidPooling
import keras.backend as K
import numpy as np
from keras.models import Sequential



class hiarBayesGoogLeNet:
    @staticmethod
    def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1,1), name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name
        else:
            bn_name = None
            conv_name = None
     
        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x
 
    @staticmethod
    def Inception(x, nb_filter, name=None):
        """
        branch1x1 = hiarBayesGoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
     
        branch3x3 = hiarBayesGoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
        branch3x3 = hiarBayesGoogLeNet.Conv2d_BN(branch3x3, nb_filter,(3,3), padding='same', strides=(1,1), name=name)
     
        branch5x5 = hiarBayesGoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1),name=name)
        branch5x5 = hiarBayesGoogLeNet.Conv2d_BN(branch5x5, nb_filter, (5,5), padding='same', strides=(1,1), name=name)
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
        branchpool = hiarBayesGoogLeNet.Conv2d_BN(branchpool, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
        """
        branch1x1 = hiarBayesGoogLeNet.Conv2d_BN(x, nb_filter[0], (1,1), padding='same', strides=(1,1), name=name+'_1x1')
     
        branch3x3 = hiarBayesGoogLeNet.Conv2d_BN(x, nb_filter[1], (1,1), padding='same', strides=(1,1), name=name+'_3x3_reduce')
        branch3x3 = hiarBayesGoogLeNet.Conv2d_BN(branch3x3, nb_filter[2],(3,3), padding='same', strides=(1,1), name=name+'_3x3')
     
        branch5x5 = hiarBayesGoogLeNet.Conv2d_BN(x, nb_filter[3], (1,1), padding='same', strides=(1,1),name=name+'5x5_reduce')
        branch5x5 = hiarBayesGoogLeNet.Conv2d_BN(branch5x5, nb_filter[4], (5,5), padding='same', strides=(1,1), name=name+'_5x5')
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
        branchpool = hiarBayesGoogLeNet.Conv2d_BN(branchpool, nb_filter[5], (1,1), padding='same', strides=(1,1), name=name+'_pool_proj')
     
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
    def build(width, height, depth, classes, pooling_regions = [1, 3], weights="imagenet"):
        assert(isinstance(classes, list), 'Must be list type.')
        assert(len(classes) == 3, 'Must be 3 elements in the list.')
        inpt = Input(shape=(width, height, depth))
        #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
        x = hiarBayesGoogLeNet.Conv2d_BN(inpt, 64, (7,7), strides=(2,2), padding='same', name="conv1_7x7_s2")
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        x = hiarBayesGoogLeNet.Conv2d_BN(x, 192, (3,3), strides=(1,1), padding='same', name="conv2_3x3")
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        """
        x = hiarBayesGoogLeNet.Inception(x, 64, name="inception_3a")#256
        x = hiarBayesGoogLeNet.Inception(x, 120, name="inception_3b")#480
        """
        x = hiarBayesGoogLeNet.Inception(x, [64,96,128,16,32,32], name="inception_3a")#256
        x = hiarBayesGoogLeNet.Inception(x, [128,128,192,32,96,64], name="inception_3b")#480
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        """
        x = hiarBayesGoogLeNet.Inception(x, 128, name="inception_4a")#512
        x = hiarBayesGoogLeNet.Inception(x, 128, name="inception_4b")
        x = hiarBayesGoogLeNet.Inception(x, 128, name="inception_4c")
        x = hiarBayesGoogLeNet.Inception(x, 132, name="inception_4d")#528
        x = hiarBayesGoogLeNet.Inception(x, 208, name="inception_4e")#832
        """
        x = hiarBayesGoogLeNet.Inception(x, [192,96,208,16,48,64], name="inception_4a")#512
        fea_low = x
        #fea_low = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv1_e')(x)
        #fea_low = GlobalAveragePooling2D()(x)#, name="gap_low"
        #fea_low = Dense(512, activation='relu')(fea_low)
        x = hiarBayesGoogLeNet.Inception(x, [160,112,224,24,64,64], name="inception_4b")
        x = hiarBayesGoogLeNet.Inception(x, [128,128,256,24,64,64], name="inception_4c")
        x = hiarBayesGoogLeNet.Inception(x, [112,144,288,32,64,64], name="inception_4d")#528
        fea_mid = x
        #fea_mid = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv2_e')(x)
        #fea_mid = GlobalAveragePooling2D()(x)#, name="gap_mid"
        #fea_mid = Dense(512, activation='relu')(fea_mid)
        x = hiarBayesGoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_4e")#832
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        """
        x = hiarBayesGoogLeNet.Inception(x, 208, name="inception_5a")
        x = hiarBayesGoogLeNet.Inception(x, 256, name="inception_5b")#1024
        """
        x = hiarBayesGoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_5a")
        x = hiarBayesGoogLeNet.Inception(x, [384,192,384,48,128,128], name="inception_5b")#1024
        fea_hig = x
        #fea_hig = Conv2D(1024, (3, 3), padding='same', activation='relu', name='conv3_e')(x)
        #fea_hig = GlobalAveragePooling2D()(x)#, name="gap_hig"
        #fea_hig = Dense(1024, activation='relu')(fea_hig)
        """
        predictions_low = Dense(classes[0], name="low", activation="sigmoid")(fea_low)#
        predictions_mid_hs = Dense(classes[1], name="middle_hs", activation="sigmoid")(fea_mid)#
        predictions_mid_ub = Dense(classes[2], name="middle_ub", activation="sigmoid")(fea_mid)#
        predictions_mid_lb = Dense(classes[3], name="middle_lb", activation="sigmoid")(fea_mid)#
        predictions_mid_sh = Dense(classes[4], name="middle_sh", activation="sigmoid")(fea_mid)#
        predictions_mid_at = Dense(classes[5], name="middle_at", activation="sigmoid")(fea_mid)#
        predictions_mid_ot = Dense(classes[6], name="middle_ot", activation="sigmoid")(fea_mid)#
        predictions_hig = Dense(classes[7], name="high_fea", activation="sigmoid")(fea_hig)#
        """
        fea_low = Conv2D(512, (3, 3), padding='same', activation='relu')(fea_low)
        #fea_low = Flatten()(fea_low)
        #fea_low = Dense(512, activation='relu')(fea_low)
        fea_low = GlobalAveragePooling2D()(fea_low)
        predictions_low = Dense(classes[0], name="low", activation="sigmoid")(fea_low)#
        fea_mid_hs = Conv2D(512, (3, 3), padding='same', activation='relu')(fea_mid)
        #fea_mid_hs = Flatten()(fea_mid_hs)
        #fea_mid_hs = Dense(512, activation='relu')(fea_mid_hs)
        fea_mid_hs = GlobalAveragePooling2D()(fea_mid_hs)
        predictions_mid_hs = Dense(classes[1], name="middle_hs", activation="sigmoid")(fea_mid_hs)#
        fea_mid_ub = Conv2D(512, (3, 3), padding='same', activation='relu')(fea_mid)
        #fea_mid_ub = Flatten()(fea_mid_ub)
        #fea_mid_ub = Dense(512, activation='relu')(fea_mid_ub)
        fea_mid_ub = GlobalAveragePooling2D()(fea_mid_ub)
        predictions_mid_ub = Dense(classes[2], name="middle_ub", activation="sigmoid")(fea_mid_ub)#
        fea_mid_lb = Conv2D(512, (3, 3), padding='same', activation='relu')(fea_mid)
        #fea_mid_lb = Flatten()(fea_mid_lb)
        #fea_mid_lb = Dense(512, activation='relu')(fea_mid_lb)
        fea_mid_lb = GlobalAveragePooling2D()(fea_mid_lb)
        predictions_mid_lb = Dense(classes[3], name="middle_lb", activation="sigmoid")(fea_mid_lb)#
        fea_mid_sh = Conv2D(512, (3, 3), padding='same', activation='relu')(fea_mid)
        #fea_mid_sh = Flatten()(fea_mid_sh)
        #fea_mid_sh = Dense(512, activation='relu')(fea_mid_sh)
        fea_mid_sh = GlobalAveragePooling2D()(fea_mid_sh)
        predictions_mid_sh = Dense(classes[4], name="middle_sh", activation="sigmoid")(fea_mid_sh)#
        fea_mid_at = Conv2D(512, (3, 3), padding='same', activation='relu')(fea_mid)
        #fea_mid_at = Flatten()(fea_mid_at)
        #fea_mid_at = Dense(512, activation='relu')(fea_mid_at)
        fea_mid_at = GlobalAveragePooling2D()(fea_mid_at)
        predictions_mid_at = Dense(classes[5], name="middle_at", activation="sigmoid")(fea_mid_at)#
        fea_mid_ot = Conv2D(512, (3, 3), padding='same', activation='relu')(fea_mid)
        #fea_mid_ot = Flatten()(fea_mid_ot)
        #fea_mid_ot = Dense(512, activation='relu')(fea_mid_ot)
        fea_mid_ot = GlobalAveragePooling2D()(fea_mid_ot)
        predictions_mid_ot = Dense(classes[6], name="middle_ot", activation="sigmoid")(fea_mid_ot)#
        fea_hig = Conv2D(1024, (3, 3), padding='same', activation='relu')(fea_hig)
        #fea_hig = Flatten()(fea_hig)
        #fea_hig = Dense(512, activation='relu')(fea_hig)
        fea_hig = GlobalAveragePooling2D()(fea_hig)
        predictions_hig = Dense(classes[7], name="high_fea", activation="sigmoid")(fea_hig)
        #"""
        """PCM2018"""
        #predictions_hig = Dense(classes[2], activation="sigmoid", name="high")(concatenate([fea_low, fea_mid, fea_hig], axis=1))
        """PCM2018"""
        predictions_priori = concatenate([predictions_low, predictions_mid_hs, predictions_mid_ub, predictions_mid_lb, predictions_mid_sh, predictions_mid_at, predictions_mid_ot], axis=1)
        """mar"""
        #val = np.load("../results/state_transition_matrix.npy")
        #state_transition_matrix = K.variable(value=val, dtype='float32', name='state_transition_matrix')
        #predictions_hig_cond = Lambda(lambda x:K.dot(x, state_transition_matrix), name="high_cond")(predictions_priori)
        """mar"""
        predictions_hig_cond = Dense(classes[7], name="high_cond", activation="sigmoid")(predictions_priori)#
        #predictions_priori = K.reshape(concatenate([predictions_low, predictions_mid], axis=1), (-1, classes[0]+classes[1], 1))
        #predictions_hig_cond = LSTM(classes[2], activation="sigmoid", name="high_cond")(predictions_priori)
        predictions_hig_posterior = Lambda(lambda x:x[1] * x[0], name="high")([predictions_hig_cond, predictions_hig])
        #predictions_hig_posterior = Lambda(lambda x:K.sigmoid(K.tanh((x[1] - 0.5) * np.pi) * x[0]), name="high")([predictions_hig_cond, predictions_hig])
        #multi#Lambda(lambda x:x[0] * x[1], name="high_post")([predictions_hig_cond, predictions_hig])
        #cond#Dense(classes[2], activation="sigmoid", name="high_post")(concatenate([predictions_hig, predictions_hig_cond], axis=1))
        #add#Lambda(lambda x:(x[0] + x[1])/2, name="high_post")([predictions_hig_cond, predictions_hig])
        """"mar"""
        #predictions_low = Activation("sigmoid")(predictions_low)
        #predictions_mid = Activation("sigmoid")(predictions_mid)
        #predictions_hig_posterior = Activation("sigmoid")(predictions_hig_posterior)
        """mar"""
        #predictions = concatenate([predictions_low, predictions_mid, predictions_hig_posterior], axis=1)
        """PCM2018"""
        #predictions = concatenate([predictions_low, predictions_mid, predictions_hig], axis=1)
        """PCM2018"""
        """
        predictions_low = Dense(classes[0], activation="sigmoid", name="low")(fea_low)
        predictions_mid_fea = Dense(classes[1], activation="sigmoid", name="middle_fea")(fea_mid)
        predictions_mid_cond = Dense(classes[1], activation="sigmoid", name="middle_cond")(predictions_low)
        predictions_mid = Lambda(lambda x:(x[0] + x[1])/2, name="mid")([predictions_mid_fea, predictions_mid_cond])
        predictions_hig_fea = Dense(classes[2], activation="sigmoid", name="high_fea")(fea_hig)
        predictions_priori = concatenate([predictions_low, predictions_mid], axis=1)
        predictions_hig_cond = Dense(classes[2], activation="sigmoid", name="high_cond")(predictions_priori)
        predictions_hig = Lambda(lambda x:(x[0] + x[1])/2, name="high_post")([predictions_hig_cond, predictions_hig_fea])
        predictions = concatenate([predictions_low, predictions_mid, predictions_hig], axis=1)
        """
        """
        x = concatenate([spp_low, spp_mid, spp_hig], axis=1)#2048
        #x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)
        x = Dropout(0.4)(x)
        x = Dense(2048, activation='relu')(x)
        x = Dense(classes, activation='softmax')(x)
        """
        # create the model
        model = Model(inpt, [predictions_low, predictions_mid_hs, predictions_mid_ub, predictions_mid_lb, predictions_mid_sh, predictions_mid_at, predictions_mid_ot, predictions_hig_posterior], name='inception')
        if weights == "imagenet":
            weights = np.load("../results/googlenet_weights.npy", encoding='latin1').item()
            for layer in model.layers:
                if layer.get_weights() == []:
                    continue
                #weight = layer.get_weights()
                if layer.name in weights:
                    #print(layer.name, end=':')
                    #print(layer.get_weights()[0].shape == weights[layer.name]['weights'].shape, end=' ')
                    #print(layer.get_weights()[1].shape == weights[layer.name]['biases'].shape)
                    layer.set_weights([weights[layer.name]['weights'], weights[layer.name]['biases']])
        # return the constructed network architecture
        return model

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    model = hiarBayesGoogLeNet.build(160, 75, 3, [10, 20, 30])#因为googleNet默认输入32*32的图片
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file="../../results/models/hiarBayesGoogleLenet.png", show_shapes=True)