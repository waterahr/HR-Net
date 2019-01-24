import sys
sys.path.append("..")
import os
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import plot_model
#from spp.spp.SpatialPyramidPooling import SpatialPyramidPooling
import keras.backend as K
import numpy as np
from keras.models import Sequential



class hiarGoogLeNet_mid:
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
        branch1x1 = hiarGoogLeNet_mid.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
     
        branch3x3 = hiarGoogLeNet_mid.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
        branch3x3 = hiarGoogLeNet_mid.Conv2d_BN(branch3x3, nb_filter,(3,3), padding='same', strides=(1,1), name=None)
     
        branch5x5 = hiarGoogLeNet_mid.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1),name=None)
        branch5x5 = hiarGoogLeNet_mid.Conv2d_BN(branch5x5, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
        branchpool = hiarGoogLeNet_mid.Conv2d_BN(branchpool, nb_filter, (1,1), padding='same', strides=(1,1), name=None)
        """
        branch1x1 = hiarGoogLeNet_mid.Conv2d_BN(x, nb_filter[0], (1,1), padding='same', strides=(1,1), name=name+'_1x1')
     
        branch3x3 = hiarGoogLeNet_mid.Conv2d_BN(x, nb_filter[1], (1,1), padding='same', strides=(1,1), name=name+'_3x3_reduce')
        branch3x3 = hiarGoogLeNet_mid.Conv2d_BN(branch3x3, nb_filter[2],(3,3), padding='same', strides=(1,1), name=name+'_3x3')
     
        branch5x5 = hiarGoogLeNet_mid.Conv2d_BN(x, nb_filter[3], (1,1), padding='same', strides=(1,1),name=name+'5x5_reduce')
        branch5x5 = hiarGoogLeNet_mid.Conv2d_BN(branch5x5, nb_filter[4], (5,5), padding='same', strides=(1,1), name=name+'_5x5')
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
        branchpool = hiarGoogLeNet_mid.Conv2d_BN(branchpool, nb_filter[5], (1,1), padding='same', strides=(1,1), name=name+'_pool_proj')
         
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
        x = hiarGoogLeNet_mid.Conv2d_BN(inpt, 64, (7,7), strides=(2,2), padding='same', name="conv1_7x7_s2")
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        x = hiarGoogLeNet_mid.Conv2d_BN(x, 192, (3,3), strides=(1,1), padding='same', name="conv2_3x3")
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        """
        x = hiarGoogLeNet.Inception(x, 64, name="inception_3a")#256
        x = hiarGoogLeNet.Inception(x, 120, name="inception_3b")#480
        """
        x = hiarGoogLeNet_mid.Inception(x, [64,96,128,16,32,32], name="inception_3a")#256
        x = hiarGoogLeNet_mid.Inception(x, [128,128,192,32,96,64], name="inception_3b")#480
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        """
        x = hiarGoogLeNet.Inception(x, 128, name="inception_4a")#512
        x = hiarGoogLeNet.Inception(x, 128, name="inception_4b")
        x = hiarGoogLeNet.Inception(x, 128, name="inception_4c")
        x = hiarGoogLeNet.Inception(x, 132, name="inception_4d")#528
        x = hiarGoogLeNet.Inception(x, 208, name="inception_4e")#832
        """
        x = hiarGoogLeNet_mid.Inception(x, [192,96,208,16,48,64], name="inception_4a")#512
        fea_low = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv1_e')(x)
        fea_low = GlobalAveragePooling2D()(fea_low)
        fea_low = Dense(512, activation='relu')(fea_low)
        x = hiarGoogLeNet_mid.Inception(x, [160,112,224,24,64,64], name="inception_4b")
        x = hiarGoogLeNet_mid.Inception(x, [128,128,256,24,64,64], name="inception_4c")
        x = hiarGoogLeNet_mid.Inception(x, [112,144,288,32,64,64], name="inception_4d")#528
        fea_mid = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv2_e')(x)
        fea_mid = GlobalAveragePooling2D()(fea_mid)
        fea_mid = Dense(512, activation='relu')(fea_mid)	
        x = hiarGoogLeNet_mid.Inception(x, [256,160,320,32,128,128], name="inception_4e")#832
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        """
        x = hiarGoogLeNet.Inception(x, 208, name="inception_5a")
        x = hiarGoogLeNet.Inception(x, 256, name="inception_5b")#1024
        """
        x = hiarGoogLeNet_mid.Inception(x, [256,160,320,32,128,128], name="inception_5a")
        x = hiarGoogLeNet_mid.Inception(x, [384,192,384,48,128,128], name="inception_5b")#1024
        fea_hig = Conv2D(1024, (3, 3), padding='same', activation='relu', name='conv3_e')(x)
        fea_hig = GlobalAveragePooling2D()(fea_hig)
        fea_hig = Dense(1024, activation='relu')(fea_hig)
        #predictions_low = Dense(classes[0], activation="sigmoid", name="low")(fea_low)
        predictions_mid = Dense(classes[1], activation="sigmoid", name="middle")(fea_mid)
        #predictions_hig = Dense(classes[2], activation="sigmoid", name="high")(fea_hig)
        #predictions = concatenate([predictions_low, predictions_mid, predictions_hig], axis=1)
        predictions = predictions_mid
        """
        x = concatenate([spp_low, spp_mid, spp_hig], axis=1)#2048
        #x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)
        x = Dropout(0.4)(x)
        x = Dense(2048, activation='relu')(x)
        x = Dense(classes, activation='softmax')(x)
        """
        # create the model
        model = Model(inpt, predictions, name='inception')
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
    model = hiarGoogLeNet_mid.build(160, 75, 3, [10, 20, 30])#因为googleNet默认输入32*32的图片
    plot_model(model, to_file="../../results/hiarGoogleLenet.png", show_shapes=True)
