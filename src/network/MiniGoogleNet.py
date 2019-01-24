# import the necessary packages
import sys
sys.path.append("..")
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K
from keras.utils import plot_model
 
class MiniGoogLeNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):#x表示输入数据，K表示conv的filter的数量，KX,KY表示kernel_size
        #define a CONV => BN => RELU pattern,我们严格按照原论文的说法，使用CONV => BN => RELU的顺序，但是实际上，CONV => Relu => BN的效果会更好一些
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        # return the block
        return x
 
    @staticmethod
    def inception_module(x,numK1_1,numK3_3,chanDim):#x表示输入数据,numK1_1,numK3_3表示kernel的filter的数量，chanDim：first_channel or last_channel
        conv1_1=MiniGoogLeNet.conv_module(x,numK1_1,1,1,(1,1),chanDim)
        conv3_3=MiniGoogLeNet.conv_module(x,numK3_3,3,3,(1,1),chanDim)
        x=concatenate([conv1_1,conv3_3],axis=chanDim)#将conv1_1和conv3_3串联到一起
        return x
 
    @staticmethod
    def downsample_module(x,K,chanDim):#K表示conv的filter的数量
        conv3_3=MiniGoogLeNet.conv_module(x,K,3,3,(2,2),chanDim,padding='valid')#padding=same表示：出输入和输出的size是相同的，由于加入了padding，如果是padding=valid，那么padding=0
        pool=MaxPooling2D((3,3),strides=(2,2))(x)
        x=concatenate([conv3_3,pool],axis=chanDim)#将conv3_3和maxPooling串到一起
        return x
 
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)#keras默认channel last，tf作为backend
        chanDim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # define the model input and first CONV module
        inputs = Input(shape=inputShape)
 
 
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1),chanDim)
        # two Inception modules followed by a downsample module
 
        x = MiniGoogLeNet.inception_module(x, 32, 32, chanDim)#第一个分叉
        x = MiniGoogLeNet.inception_module(x, 32, 48, chanDim)#第二个分叉
        x = MiniGoogLeNet.downsample_module(x, 80, chanDim)#第三个分叉，含有maxpooling
 
 
 
        # four Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x, 112, 48, chanDim)
        x = MiniGoogLeNet.inception_module(x, 96, 64, chanDim)
        x = MiniGoogLeNet.inception_module(x, 80, 80, chanDim)
        x = MiniGoogLeNet.inception_module(x, 48, 96, chanDim)
        x = MiniGoogLeNet.downsample_module(x, 96, chanDim)
 
        # two Inception modules followed by global POOL and dropout
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)#输出是（7×7×（160+176））
        x = AveragePooling2D((7, 7))(x)#经过平均池化之后变成了（1*1*376）
        x = Dropout(0.5)(x)
 
        # softmax classifier
        x = Flatten()(x)#特征扁平化
        x = Dense(classes)(x)#全连接层，进行多分类,形成最终的10分类
        x = Activation("softmax")(x)
        # create the model
        model = Model(inputs, x, name="googlenet")
        # return the constructed network architecture
        return model

if __name__ == "__main__":
    model = MiniGoogLeNet.build(32, 32, 3, 10)#因为googleNet默认输入32*32的图片
    plot_model(model, to_file="../../results/miniGoogleLenet.png", show_shapes=True)
