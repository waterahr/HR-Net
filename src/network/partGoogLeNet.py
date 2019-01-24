import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import sys
sys.path.append("..")
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate
from keras.utils import plot_model



class partGoogLeNet:
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
        branch1x1 = partGoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
     
        branch3x3 = partGoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
        branch3x3 = partGoogLeNet.Conv2d_BN(branch3x3, nb_filter,(3,3), padding='same', strides=(1,1), name=name)
     
        branch5x5 = partGoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1),name=name)
        branch5x5 = partGoogLeNet.Conv2d_BN(branch5x5, nb_filter, (5,5), padding='same', strides=(1,1), name=name)
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
        branchpool = partGoogLeNet.Conv2d_BN(branchpool, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
        """
        branch1x1 = partGoogLeNet.Conv2d_BN(x, nb_filter[0], (1,1), padding='same', strides=(1,1), name=name+'_1x1', trainable=trainable)
     
        branch3x3 = partGoogLeNet.Conv2d_BN(x, nb_filter[1], (1,1), padding='same', strides=(1,1), name=name+'_3x3_reduce', trainable=trainable)
        branch3x3 = partGoogLeNet.Conv2d_BN(branch3x3, nb_filter[2],(3,3), padding='same', strides=(1,1), name=name+'_3x3', trainable=trainable)
     
        branch5x5 = partGoogLeNet.Conv2d_BN(x, nb_filter[3], (1,1), padding='same', strides=(1,1),name=name+'5x5_reduce', trainable=trainable)
        branch5x5 = partGoogLeNet.Conv2d_BN(branch5x5, nb_filter[4], (5,5), padding='same', strides=(1,1), name=name+'_5x5', trainable=trainable)
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', trainable=trainable)(x)
        branchpool = partGoogLeNet.Conv2d_BN(branchpool, nb_filter[5], (1,1), padding='same', strides=(1,1), name=name+'_pool_proj', trainable=trainable)
     
        x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)
     
        return x
    

    @staticmethod
    def build(width, height, depth, classes, weights="imagenet", model_depth=9):
        attributes_list = ['accessoryHeadphone', 'personalLess15', 'personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'carryingBabyBuggy', 'carryingBackpack', 'hairBald', 'footwearBoots', 'lowerBodyCapri', 'carryingOther', 'carryingShoppingTro', 'carryingUmbrella', 'lowerBodyCasual', 'upperBodyCasual', 'personalFemale', 'carryingFolder', 'lowerBodyFormal', 'upperBodyFormal', 'accessoryHairBand', 'accessoryHat', 'lowerBodyHotPants', 'upperBodyJacket', 'lowerBodyJeans', 'accessoryKerchief', 'footwearLeatherShoes', 'upperBodyLogo', 'hairLong', 'lowerBodyLongSkirt', 'upperBodyLongSleeve', 'lowerBodyPlaid', 'lowerBodyThinStripes', 'carryingLuggageCase', 'personalMale', 'carryingMessengerBag', 'accessoryMuffler', 'accessoryNothing', 'carryingNothing', 'upperBodyNoSleeve', 'upperBodyPlaid', 'carryingPlasticBags', 'footwearSandals', 'footwearShoes', 'hairShort', 'lowerBodyShorts', 'upperBodyShortSleeve', 'lowerBodyShortSkirt', 'footwearSneakers', 'footwearStocking', 'upperBodyThinStripes', 'upperBodySuit', 'carryingSuitcase', 'lowerBodySuits', 'accessorySunglasses', 'upperBodySweater', 'upperBodyThickStripes', 'lowerBodyTrousers', 'upperBodyTshirt', 'upperBodyOther', 'upperBodyVNeck', 'footwear', 'hair', 'lowerbody', 'upperbody']
        attributes_list = np.asarray(attributes_list)
        for i in range(len(classes)):
            print("-----------------")
            print(attributes_list[classes[i]])
            print("-----------------")
        inpt = Input(shape=(width, height, depth))
        #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
        x = partGoogLeNet.Conv2d_BN(inpt, 64, (7,7), strides=(2,2), padding='same', name="conv1_7x7_s2", trainable=False)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=False)(x)
        x = partGoogLeNet.Conv2d_BN(x, 192, (3,3), strides=(1,1), padding='same', name="conv2_3x3", trainable=False)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=False)(x)
        """
        x = partGoogLeNet.Inception(x, 64, name="inception_3a")#256
        x = partGoogLeNet.Inception(x, 120, name="inception_3b")#480
        """
        x = partGoogLeNet.Inception(x, [64,96,128,16,32,32], name="inception_3a", trainable=False)#256
        if model_depth==1: fea = x
        x = partGoogLeNet.Inception(x, [128,128,192,32,96,64], name="inception_3b", trainable=False)#480
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=False)(x)
        if model_depth==2: fea = x
        """
        x = partGoogLeNet.Inception(x, 128, name="inception_4a")#512
        x = partGoogLeNet.Inception(x, 128, name="inception_4b")
        x = partGoogLeNet.Inception(x, 128, name="inception_4c")
        x = partGoogLeNet.Inception(x, 132, name="inception_4d")#528
        x = partGoogLeNet.Inception(x, 208, name="inception_4e")#832
        """
        x = partGoogLeNet.Inception(x, [192,96,208,16,48,64], name="inception_4a", trainable=False)#512
        if model_depth==3: fea = x
        x = partGoogLeNet.Inception(x, [160,112,224,24,64,64], name="inception_4b", trainable=False)
        if model_depth==4: fea = x
        x = partGoogLeNet.Inception(x, [128,128,256,24,64,64], name="inception_4c", trainable=False)
        if model_depth==5: fea = x
        x = partGoogLeNet.Inception(x, [112,144,288,32,64,64], name="inception_4d", trainable=False)#528
        if model_depth==6: fea = x
        x = partGoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_4e", trainable=False)#832
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=False)(x)
        if model_depth==7: fea = x
        """
        x = partGoogLeNet.Inception(x, 208, name="inception_5a")
        x = partGoogLeNet.Inception(x, 256, name="inception_5b")#1024
        """
        x = partGoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_5a", trainable=False)
        if model_depth==8: fea = x
        x = partGoogLeNet.Inception(x, [384,192,384,48,128,128], name="inception_5b", trainable=False)#1024
        if model_depth==9: fea = x
        #x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)
        #x = GlobalAveragePooling2D()(x)
        x = GlobalAveragePooling2D()(fea)
        #x = Dropout(0.4)(x)
        name = "dense_"
        if model_depth != 9:
            name += "_"+str(model_depth)+"_"
        predictions = []
        for i in range(len(classes)):
            tmp = Dense(len(classes[i]), activation='sigmoid', name=name+str(i))(x)
            predictions.append(tmp)
        predictions = concatenate(predictions, axis=1)
        # create the model
        model = Model(inpt, predictions, name='inception')
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
    model = partGoogLeNet.build(32, 32, 3, [[1],[2],[3],[4]])#因为googleNet默认输入32*32的图片
    model.summary()
    plot_model(model, to_file="../results/partGoogleLenet.png", show_shapes=True)
