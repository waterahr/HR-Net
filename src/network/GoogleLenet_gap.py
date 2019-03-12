import sys
sys.path.append("..")
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate
from keras.layers.core import Flatten
from keras.utils import plot_model



class GoogLeNet:
    @staticmethod
    def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1,1), name=None, trainable=True):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name
        else:
            bn_name = None
            conv_name = None
     
        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name, trainable=trainable)(x)
        x = BatchNormalization(axis=3, name=bn_name, trainable=trainable)(x)
        return x
 
    @staticmethod
    def Inception(x, nb_filter, name=None, trainable=True):
        """
        branch1x1 = GoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
     
        branch3x3 = GoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
        branch3x3 = GoogLeNet.Conv2d_BN(branch3x3, nb_filter,(3,3), padding='same', strides=(1,1), name=name)
     
        branch5x5 = GoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1),name=name)
        branch5x5 = GoogLeNet.Conv2d_BN(branch5x5, nb_filter, (5,5), padding='same', strides=(1,1), name=name)
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
        branchpool = GoogLeNet.Conv2d_BN(branchpool, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
        """
        branch1x1 = GoogLeNet.Conv2d_BN(x, nb_filter[0], (1,1), padding='same', strides=(1,1), name=name+'_1x1', trainable=trainable)
     
        branch3x3 = GoogLeNet.Conv2d_BN(x, nb_filter[1], (1,1), padding='same', strides=(1,1), name=name+'_3x3_reduce', trainable=trainable)
        branch3x3 = GoogLeNet.Conv2d_BN(branch3x3, nb_filter[2],(3,3), padding='same', strides=(1,1), name=name+'_3x3', trainable=trainable)
     
        branch5x5 = GoogLeNet.Conv2d_BN(x, nb_filter[3], (1,1), padding='same', strides=(1,1),name=name+'5x5_reduce', trainable=trainable)
        branch5x5 = GoogLeNet.Conv2d_BN(branch5x5, nb_filter[4], (5,5), padding='same', strides=(1,1), name=name+'_5x5', trainable=trainable)
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', trainable=trainable)(x)
        branchpool = GoogLeNet.Conv2d_BN(branchpool, nb_filter[5], (1,1), padding='same', strides=(1,1), name=name+'_pool_proj', trainable=trainable)
     
        x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)
     
        return x
    

    @staticmethod
    def build(width, height, depth, classes, weights="imagenet", model_depth=9):
        inpt = Input(shape=(width, height, depth))
        #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
        x = GoogLeNet.Conv2d_BN(inpt, 64, (7,7), strides=(2,2), padding='same', name="conv1_7x7_s2", trainable=True)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        x = GoogLeNet.Conv2d_BN(x, 192, (3,3), strides=(1,1), padding='same', name="conv2_3x3", trainable=True)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        """
        x = GoogLeNet.Inception(x, 64, name="inception_3a")#256
        x = GoogLeNet.Inception(x, 120, name="inception_3b")#480
        """
        x = GoogLeNet.Inception(x, [64,96,128,16,32,32], name="inception_3a", trainable=True)#256
        if model_depth==1: fea = x
        x = GoogLeNet.Inception(x, [128,128,192,32,96,64], name="inception_3b", trainable=True)#480
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        if model_depth==2: fea = x
        """
        x = GoogLeNet.Inception(x, 128, name="inception_4a")#512
        x = GoogLeNet.Inception(x, 128, name="inception_4b")
        x = GoogLeNet.Inception(x, 128, name="inception_4c")
        x = GoogLeNet.Inception(x, 132, name="inception_4d")#528
        x = GoogLeNet.Inception(x, 208, name="inception_4e")#832
        """
        x = GoogLeNet.Inception(x, [192,96,208,16,48,64], name="inception_4a", trainable=True)#512
        if model_depth==3: fea = x
        x = GoogLeNet.Inception(x, [160,112,224,24,64,64], name="inception_4b", trainable=True)
        if model_depth==4: fea = x
        x = GoogLeNet.Inception(x, [128,128,256,24,64,64], name="inception_4c", trainable=True)
        if model_depth==5: fea = x
        x = GoogLeNet.Inception(x, [112,144,288,32,64,64], name="inception_4d", trainable=True)#528
        if model_depth==6: fea = x
        x = GoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_4e", trainable=True)#832
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        if model_depth==7: fea = x
        """
        x = GoogLeNet.Inception(x, 208, name="inception_5a")
        x = GoogLeNet.Inception(x, 256, name="inception_5b")#1024
        """
        x = GoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_5a", trainable=True)
        if model_depth==8: fea = x
        x = GoogLeNet.Inception(x, [384,192,384,48,128,128], name="inception_5b", trainable=True)#1024
        if model_depth==9: fea = x
        #x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)
        #x = GlobalAveragePooling2D()(x)
        x = GlobalAveragePooling2D()(fea)
        #x = Flatten()(fea)
        #x = Dropout(0.4)(x)
        name1 = "dense_1"
        name2 = "dense_2"
        if model_depth != 9:
            name1 += "_"+str(model_depth)
            name2 += "_"+str(model_depth)
        #x = Dense(1000, activation='linear', name=name1)(x)
        x = Dense(classes, activation='sigmoid', name=name2)(x)
        # create the model
        model = Model(inpt, x, name='inception')
        # return the constructed network architecture
        if weights == "imagenet":
            print("ImageNet...")
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

        return model

if __name__ == "__main__":
    model = GoogLeNet.build(32, 32, 3, 10)#因为googleNet默认输入32*32的图片
    plot_model(model, to_file="../../results/GoogleLenet.png", show_shapes=True)
