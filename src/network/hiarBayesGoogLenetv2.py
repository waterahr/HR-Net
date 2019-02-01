import sys
sys.path.append("..")
import os
from keras.models import Model
from keras.layers import Activation, Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, GlobalAveragePooling2D, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Flatten
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
        fea_low = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv1_e')(x)
        #fea_low = GlobalAveragePooling2D()(fea_low)#, name="gap_low"
        fea_low = Flatten()(fea_low)
        fea_low = Dense(512, activation='relu')(fea_low)
        x = hiarBayesGoogLeNet.Inception(x, [160,112,224,24,64,64], name="inception_4b")
        x = hiarBayesGoogLeNet.Inception(x, [128,128,256,24,64,64], name="inception_4c")
        x = hiarBayesGoogLeNet.Inception(x, [112,144,288,32,64,64], name="inception_4d")#528
        fea_mid = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv2_e')(x)
        #fea_mid = GlobalAveragePooling2D()(fea_mid)#, name="gap_mid"
        fea_mid = Flatten()(fea_mid)
        fea_mid = Dense(512, activation='relu')(fea_mid)
        x = hiarBayesGoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_4e")#832
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        """
        x = hiarBayesGoogLeNet.Inception(x, 208, name="inception_5a")
        x = hiarBayesGoogLeNet.Inception(x, 256, name="inception_5b")#1024
        """
        x = hiarBayesGoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_5a")
        x = hiarBayesGoogLeNet.Inception(x, [384,192,384,48,128,128], name="inception_5b")#1024
        fea_hig = Conv2D(1024, (3, 3), padding='same', activation='relu', name='conv3_e')(x)
        #fea_hig = GlobalAveragePooling2D()(fea_hig)#, name="gap_hig"
        fea_hig = Flatten()(fea_hig)
        fea_hig = Dense(1024, activation='relu')(fea_hig)
        #"""
        """
        predictions_low = Dense(classes[0], name="low", activation="sigmoid")(fea_low)#
        predictions_mid = Dense(classes[1], name="middle", activation="sigmoid")(fea_mid)#
        predictions_hig = Dense(classes[2], name="high", activation="sigmoid")(fea_hig)#
        """
        pred1 = Dense(1, activation='sigmoid', name="low"+str(1))(fea_low)
        pred2 = Dense(1, activation='sigmoid', name="middle"+str(2))(fea_mid)
        pred3 = Dense(1, activation='sigmoid', name="middle"+str(3))(fea_mid)
        pred4 = Dense(1, activation='sigmoid', name="middle"+str(4))(fea_mid)
        pred5 = Dense(1, activation='sigmoid', name="middle"+str(5))(fea_mid)
        pred6 = Dense(1, activation='sigmoid', name="middle"+str(6))(fea_mid)
        pred7 = Dense(1, activation='sigmoid', name="middle"+str(7))(fea_mid)
        pred8 = Dense(1, activation='sigmoid', name="middle"+str(8))(fea_mid)
        pred9 = Dense(1, activation='sigmoid', name="middle"+str(9))(fea_mid)
        pred10 = Dense(1, activation='sigmoid', name="middle"+str(10))(fea_mid)
        pred11 = Dense(1, activation='sigmoid', name="middle"+str(11))(fea_mid)
        pred12 = Dense(1, activation='sigmoid', name="middle"+str(12))(fea_mid)
        pred13 = Dense(1, activation='sigmoid', name="middle"+str(13))(fea_mid)
        pred14 = Dense(1, activation='sigmoid', name="middle"+str(14))(fea_mid)
        pred15 = Dense(1, activation='sigmoid', name="middle"+str(15))(fea_mid)
        pred16 = Dense(1, activation='sigmoid', name="middle"+str(16))(fea_mid)
        pred17 = Dense(1, activation='sigmoid', name="middle"+str(17))(fea_mid)
        pred18 = Dense(1, activation='sigmoid', name="middle"+str(18))(fea_mid)
        pred19 = Dense(1, activation='sigmoid', name="middle"+str(19))(fea_mid)
        pred20 = Dense(1, activation='sigmoid', name="middle"+str(20))(fea_mid)
        pred21 = Dense(1, activation='sigmoid', name="middle"+str(21))(fea_mid)
        pred22 = Dense(1, activation='sigmoid', name="middle"+str(22))(fea_mid)
        pred23 = Dense(1, activation='sigmoid', name="middle"+str(23))(fea_mid)
        pred24 = Dense(1, activation='sigmoid', name="middle"+str(24))(fea_mid)
        pred25 = Dense(1, activation='sigmoid', name="middle"+str(25))(fea_mid)
        pred26 = Dense(1, activation='sigmoid', name="middle"+str(26))(fea_mid)
        pred27 = Dense(1, activation='sigmoid', name="middle"+str(27))(fea_mid)
        pred28 = Dense(1, activation='sigmoid', name="middle"+str(28))(fea_mid)
        pred29 = Dense(1, activation='sigmoid', name="middle"+str(29))(fea_mid)
        pred30 = Dense(1, activation='sigmoid', name="middle"+str(30))(fea_mid)
        pred31 = Dense(1, activation='sigmoid', name="middle"+str(31))(fea_mid)
        pred32 = Dense(1, activation='sigmoid', name="middle"+str(32))(fea_mid)
        pred33 = Dense(1, activation='sigmoid', name="middle"+str(33))(fea_mid)
        pred34 = Dense(1, activation='sigmoid', name="middle"+str(34))(fea_mid)
        pred35 = Dense(1, activation='sigmoid', name="high"+str(35))(fea_hig)
        pred36 = Dense(1, activation='sigmoid', name="high"+str(36))(fea_hig)
        pred37 = Dense(1, activation='sigmoid', name="high"+str(37))(fea_hig)
        pred38 = Dense(1, activation='sigmoid', name="high"+str(38))(fea_hig)
        pred39 = Dense(1, activation='sigmoid', name="high"+str(39))(fea_hig)
        pred40 = Dense(1, activation='sigmoid', name="high"+str(40))(fea_hig)
        pred41 = Dense(1, activation='sigmoid', name="high"+str(41))(fea_hig)
        pred42 = Dense(1, activation='sigmoid', name="high"+str(42))(fea_hig)
        pred43 = Dense(1, activation='sigmoid', name="high"+str(43))(fea_hig)
        pred44 = Dense(1, activation='sigmoid', name="high"+str(44))(fea_hig)
        pred45 = Dense(1, activation='sigmoid', name="high"+str(45))(fea_hig)
        pred46 = Dense(1, activation='sigmoid', name="high"+str(46))(fea_hig)
        pred47 = Dense(1, activation='sigmoid', name="high"+str(47))(fea_hig)
        pred48 = Dense(1, activation='sigmoid', name="high"+str(48))(fea_hig)
        pred49 = Dense(1, activation='sigmoid', name="high"+str(49))(fea_hig)
        pred50 = Dense(1, activation='sigmoid', name="high"+str(50))(fea_hig)
        pred51 = Dense(1, activation='sigmoid', name="high"+str(51))(fea_hig)
        predictions_priori = concatenate([pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9,pred10,pred11,pred12,pred13,pred14,pred15,pred16,pred17,pred18,pred19,pred20,pred21,pred22,pred23,pred24,pred25,pred26,pred27,pred28,pred29,pred30,pred31,pred32,pred33,pred34], axis=1)
        """PCM2018"""
        #predictions_hig = Dense(classes[2], activation="sigmoid", name="high")(concatenate([fea_low, fea_mid, fea_hig], axis=1))
        """PCM2018"""
        #predictions_priori = concatenate([predictions_low, predictions_mid], axis=1)
        """mar"""
        #val = np.load("../results/state_transition_matrix.npy")
        #state_transition_matrix = K.variable(value=val, dtype='float32', name='state_transition_matrix')
        #predictions_hig_cond = Lambda(lambda x:K.dot(x, state_transition_matrix), name="high_cond")(predictions_priori)
        """mar"""
        #predictions_hig_cond = Dense(classes[2], activation="sigmoid", name="high_cond")(predictions_priori)
        pred35cond = Dense(1, activation="sigmoid", name="high_cond"+str(35))(predictions_priori)
        pred36cond = Dense(1, activation="sigmoid", name="high_cond"+str(36))(predictions_priori)
        pred37cond = Dense(1, activation="sigmoid", name="high_cond"+str(37))(predictions_priori)
        pred38cond = Dense(1, activation="sigmoid", name="high_cond"+str(38))(predictions_priori)
        pred39cond = Dense(1, activation="sigmoid", name="high_cond"+str(39))(predictions_priori)
        pred40cond = Dense(1, activation="sigmoid", name="high_cond"+str(40))(predictions_priori)
        pred41cond = Dense(1, activation="sigmoid", name="high_cond"+str(41))(predictions_priori)
        pred42cond = Dense(1, activation="sigmoid", name="high_cond"+str(42))(predictions_priori)
        pred43cond = Dense(1, activation="sigmoid", name="high_cond"+str(43))(predictions_priori)
        pred44cond = Dense(1, activation="sigmoid", name="high_cond"+str(44))(predictions_priori)
        pred45cond = Dense(1, activation="sigmoid", name="high_cond"+str(45))(predictions_priori)
        pred46cond = Dense(1, activation="sigmoid", name="high_cond"+str(46))(predictions_priori)
        pred47cond = Dense(1, activation="sigmoid", name="high_cond"+str(47))(predictions_priori)
        pred48cond = Dense(1, activation="sigmoid", name="high_cond"+str(48))(predictions_priori)
        pred49cond = Dense(1, activation="sigmoid", name="high_cond"+str(49))(predictions_priori)
        pred50cond = Dense(1, activation="sigmoid", name="high_cond"+str(50))(predictions_priori)
        pred51cond = Dense(1, activation="sigmoid", name="high_cond"+str(51))(predictions_priori)
        #predictions_priori = K.reshape(concatenate([predictions_low, predictions_mid], axis=1), (-1, classes[0]+classes[1], 1))
        #predictions_hig_cond = LSTM(classes[2], activation="sigmoid", name="high_cond")(predictions_priori)
        #predictions_hig_posterior = Lambda(lambda x:x[1] * x[0], name="high_post")([predictions_hig_cond, predictions_hig])
        pred35pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(35))([pred35,pred35cond])
        pred36pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(36))([pred36,pred36cond])
        pred37pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(37))([pred37,pred37cond])
        pred38pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(38))([pred38,pred38cond])
        pred39pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(39))([pred39,pred39cond])
        pred40pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(40))([pred40,pred40cond])
        pred41pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(41))([pred41,pred41cond])
        pred42pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(42))([pred42,pred42cond])
        pred43pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(43))([pred43,pred43cond])
        pred44pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(44))([pred44,pred44cond])
        pred45pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(45))([pred45,pred45cond])
        pred46pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(46))([pred46,pred46cond])
        pred47pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(47))([pred47,pred47cond])
        pred48pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(48))([pred48,pred48cond])
        pred49pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(49))([pred49,pred49cond])
        pred50pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(50))([pred50,pred50cond])
        pred51pos = Lambda(lambda x:x[1] * x[0], name="high_post"+str(51))([pred51,pred51cond])
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
        model = Model(inpt, [pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9,pred10,pred11,pred12,pred13,pred14,pred15,pred16,pred17,pred18,pred19,pred20,pred21,pred22,pred23,pred24,pred25,pred26,pred27,pred28,pred29,pred30,pred31,pred32,pred33,pred34,pred35pos,pred36pos,pred37pos,pred38pos,pred39pos,pred40pos,pred41pos,pred42pos,pred43pos,pred44pos,pred45pos,pred46pos,pred47pos,pred48pos,pred49pos,pred50pos,pred51pos], name='inception')
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