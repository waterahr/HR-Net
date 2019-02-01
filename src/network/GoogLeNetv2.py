import sys
import os
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
        #x = GlobalAveragePooling2D()(fea)
        x = Flatten()(fea)
        x = Dropout(0.4)(x)
        name1 = "dense_1"
        name2 = "dense_2_"
        if model_depth != 9:
            name1 += "_"+str(model_depth)
            name2 += "_"+str(model_depth)
        #x = Dense(1000, activation='linear', name=name1)(x)
        x = Dense(1000, activation='relu', name=name1)(x)###dropout&flat&nolinear better
        #x = Dense(classes, activation='sigmoid', name=name2)(x)
        """
        pred = Dense(1, activation='sigmoid', name=name2+str(0))(x)
        for i in range(1, classes):
            tmp = Dense(1, activation='sigmoid', name=name2+str(i))(x)
            pred = concatenate([pred, tmp], axis=1)
        x = pred
        """
        #"""
        pred1 = Dense(1, activation='sigmoid', name=name2+str(1))(x)
        pred2 = Dense(1, activation='sigmoid', name=name2+str(2))(x)
        pred3 = Dense(1, activation='sigmoid', name=name2+str(3))(x)
        pred4 = Dense(1, activation='sigmoid', name=name2+str(4))(x)
        pred5 = Dense(1, activation='sigmoid', name=name2+str(5))(x)
        pred6 = Dense(1, activation='sigmoid', name=name2+str(6))(x)
        pred7 = Dense(1, activation='sigmoid', name=name2+str(7))(x)
        pred8 = Dense(1, activation='sigmoid', name=name2+str(8))(x)
        pred9 = Dense(1, activation='sigmoid', name=name2+str(9))(x)
        pred10 = Dense(1, activation='sigmoid', name=name2+str(10))(x)
        pred11 = Dense(1, activation='sigmoid', name=name2+str(11))(x)
        pred12 = Dense(1, activation='sigmoid', name=name2+str(12))(x)
        pred13 = Dense(1, activation='sigmoid', name=name2+str(13))(x)
        pred14 = Dense(1, activation='sigmoid', name=name2+str(14))(x)
        pred15 = Dense(1, activation='sigmoid', name=name2+str(15))(x)
        pred16 = Dense(1, activation='sigmoid', name=name2+str(16))(x)
        pred17 = Dense(1, activation='sigmoid', name=name2+str(17))(x)
        pred18 = Dense(1, activation='sigmoid', name=name2+str(18))(x)
        pred19 = Dense(1, activation='sigmoid', name=name2+str(19))(x)
        pred20 = Dense(1, activation='sigmoid', name=name2+str(20))(x)
        pred21 = Dense(1, activation='sigmoid', name=name2+str(21))(x)
        pred22 = Dense(1, activation='sigmoid', name=name2+str(22))(x)
        pred23 = Dense(1, activation='sigmoid', name=name2+str(23))(x)
        pred24 = Dense(1, activation='sigmoid', name=name2+str(24))(x)
        pred25 = Dense(1, activation='sigmoid', name=name2+str(25))(x)
        pred26 = Dense(1, activation='sigmoid', name=name2+str(26))(x)
        pred27 = Dense(1, activation='sigmoid', name=name2+str(27))(x)
        pred28 = Dense(1, activation='sigmoid', name=name2+str(28))(x)
        pred29 = Dense(1, activation='sigmoid', name=name2+str(29))(x)
        pred30 = Dense(1, activation='sigmoid', name=name2+str(30))(x)
        pred31 = Dense(1, activation='sigmoid', name=name2+str(31))(x)
        pred32 = Dense(1, activation='sigmoid', name=name2+str(32))(x)
        pred33 = Dense(1, activation='sigmoid', name=name2+str(33))(x)
        pred34 = Dense(1, activation='sigmoid', name=name2+str(34))(x)
        pred35 = Dense(1, activation='sigmoid', name=name2+str(35))(x)
        pred36 = Dense(1, activation='sigmoid', name=name2+str(36))(x)
        pred37 = Dense(1, activation='sigmoid', name=name2+str(37))(x)
        pred38 = Dense(1, activation='sigmoid', name=name2+str(38))(x)
        pred39 = Dense(1, activation='sigmoid', name=name2+str(39))(x)
        pred40 = Dense(1, activation='sigmoid', name=name2+str(40))(x)
        pred41 = Dense(1, activation='sigmoid', name=name2+str(41))(x)
        pred42 = Dense(1, activation='sigmoid', name=name2+str(42))(x)
        pred43 = Dense(1, activation='sigmoid', name=name2+str(43))(x)
        pred44 = Dense(1, activation='sigmoid', name=name2+str(44))(x)
        pred45 = Dense(1, activation='sigmoid', name=name2+str(45))(x)
        pred46 = Dense(1, activation='sigmoid', name=name2+str(46))(x)
        pred47 = Dense(1, activation='sigmoid', name=name2+str(47))(x)
        pred48 = Dense(1, activation='sigmoid', name=name2+str(48))(x)
        pred49 = Dense(1, activation='sigmoid', name=name2+str(49))(x)
        pred50 = Dense(1, activation='sigmoid', name=name2+str(50))(x)
        pred51 = Dense(1, activation='sigmoid', name=name2+str(51))(x)
        #"""
        # create the model
        #model = Model(inpt, x, name='inception')
        model = Model(inpt, [pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9,pred10,pred11,pred12,pred13,pred14,pred15,pred16,pred17,pred18,pred19,pred20,pred21,pred22,pred23,pred24,pred25,pred26,pred27,pred28,pred29,pred30,pred31,pred32,pred33,pred34,pred35,pred36,pred37,pred38,pred39,pred40,pred41,pred42,pred43,pred44,pred45,pred46,pred47,pred48,pred49,pred50,pred51], name='inception')
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
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    model = GoogLeNet.build(32, 32, 3, 10, weights="None")#因为googleNet默认输入32*32的图片
    model.summary()
    plot_model(model, to_file="../../results/GoogleLenetv2.png", show_shapes=True)
