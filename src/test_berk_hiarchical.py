"""
python test_berk_hiarchical.py -m hiarBayesGoogLeNet -w ../models/imagenet_models/hiarBayesGoogLeNet_berk/binary61_multi_epoch25_valloss0.45.hdf5
"""
from network.hiarGoogLenetSPP import hiarGoogLeNetSPP
from network.hiarGoogLenetWAM import hiarGoogLeNetWAM
from network.hiarGoogLenet import hiarGoogLeNet
from network.hiarGoogLenet_high import hiarGoogLeNet_high
from network.hiarGoogLenet_mid import hiarGoogLeNet_mid
from network.hiarGoogLenet_low import hiarGoogLeNet_low
from network.hiarBayesGoogLenet import hiarBayesGoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
from keras import backend as K
from angular_losses import bayes_binary_crossentropy

alpha = []

def parse_arg():
    models = ['hiarGoogLeNetSPP', 'hiarGoogLeNetWAM', 'hiarGoogLeNet', 'hiarBayesGoogLeNet', 'hiarGoogLeNet_high', 'hiarGoogLeNet_mid', 'hiarGoogLeNet_low']
    parser = argparse.ArgumentParser(description='training of the WPAL...')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='The gpu device\'s ID need to be used')
    parser.add_argument('-c', '--classes', type=int, default=9,
                        help='The total number of classes to be predicted')
    parser.add_argument('-b', '--batch', type=int, default=64,
                        help='The batch size of the training process')
    parser.add_argument('-wd', '--width', type=int, default=420,
                        help='The width of thWPAL_berke picture')
    parser.add_argument('-hg', '--height', type=int, default=210,
                        help='The height of the picture')
    parser.add_argument('-w', '--weight', type=str, default='',
                        help='The weights file of the pre-training')
    parser.add_argument('-m', '--model', type=str, default='',
                        help='The model including: '+str(models))
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args


    


if __name__ == "__main__":
    #"""
    save_name = "binary9"
    low_level = [1]#, 61, 62, 63, 64
    mid_level = [2,3,4,5,6,7,8]
    high_level = [0]
    alpha = np.load("../results/relation_array.npy")
    """
    save_name = "binary39"
    low_level = [27, 32, 50, 56]
    mid_level = [7, 11, 21, 23, 24, 26, 28, 29, 35, 36, 37, 38, 41, 42, 43, 45, 46, 47, 48, 54, 57, 58, 59, 60]
    high_level = [2, 3, 4, 5, 14, 15, 18, 19, 31, 34, 40]
    """
    args = parse_arg()
    class_num = args.classes
    alpha = np.zeros((class_num,))


    # Data augmentation to pre-processing
    heavy_augmentation = True
    if heavy_augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=45,
            width_shift_range=0.25,
            height_shift_range=0.25,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.5,
            channel_shift_range=0.5,
            fill_mode='nearest')
    else:
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.125,
            height_shift_range=0.125,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest')
    image_width = args.width
    image_height = args.height
    #hiarBayesGoogLeNet
    if args.model == "hiarGoogLeNetSPP":
        filename = r"../results/berk-test.csv"
    elif args.model == "hiarGoogLeNetWAM":
        filename = r"../results/berk-test.csv"
    elif args.model == "hiarGoogLeNet":
        filename = r"../results/berk-test.csv"
    elif args.model == "hiarGoogLeNet_high":
        filename = r"../results/berk-test.csv"
    elif args.model == "hiarGoogLeNet_mid":
        filename = r"../results/berk-test.csv"
    elif args.model == "hiarGoogLeNet_low":
        filename = r"../results/berk-test.csv"
    elif args.model == "hiarBayesGoogLeNet":
        filename = r"../results/berk-test.csv"
    data = np.array(pd.read_csv(filename))[:, 1:]
    length = len(data)
    #global alpha
    data_x = np.zeros((length, image_width, image_height, 3))
    data_y = np.zeros((length, class_num))
    for i in range(length):
        #img = image.load_img(path + m)
        img = image.load_img(data[i, 0], target_size=(image_width, image_height, 3))
        data_x[i] = image.img_to_array(img)
        data_y[i] = np.array(data[i, 1:1+class_num], dtype="float32")
    data_y = data_y[:, list(np.hstack((low_level, mid_level, high_level)))]
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)
    X_test = data_x
    y_test = data_y
    if args.model == "hiarGoogLeNet_high":
        y_test = y_test[:, len(low_level)+len(mid_level):]
    elif args.model == "hiarGoogLeNet_mid":
        y_test = y_test[:, len(low_level):len(low_level)+len(mid_level)]
    elif args.model == "hiarGoogLeNet_low":
        y_test = y_test[:, :len(low_level)]
    print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    
    
    #googleNet默认输入32*32的图片
    if args.model == "hiarGoogLeNetSPP":
        model = hiarGoogLeNetSPP.build(None, None, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarGoogLeNetWAM":
        model = hiarGoogLeNetWAM.build(None, None, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarGoogLeNet":
        model = hiarGoogLeNet.build(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarGoogLeNet_high":
        model = hiarGoogLeNet_high.build(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarGoogLeNet_mid":
        model = hiarGoogLeNet_mid.build(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarGoogLeNet_low":
        model = hiarGoogLeNet_low.build(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarBayesGoogLeNet":
        model = hiarBayesGoogLeNet.build(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    gpus_num = len(args.gpus.split(','))
    if gpus_num > 1:
        multi_gpu_model(model, gpus=gpus_num)
    #model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model.compile(loss=loss_func, optimizer='adam', loss_weights=loss_weights, metrics=metrics)
    model.summary()
    model.load_weights(args.weight, by_name=True)
    
    predictions = model.predict(X_test)
    print("The shape of the predictions_test is: ", predictions.shape)
    np.save("../results/predictions/" + args.model + '_' + save_name + "_predictions500_imagenet_berk.npy", predictions)