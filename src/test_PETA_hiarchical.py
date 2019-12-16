"""
python test_PETA_hiarchical.py -g 0 -c 61 -m hiarGoogLeNetSPP -w ../models/
python test_PETA_hiarchical.py -g 0 -m hiarGoogLeNet -c 61 -w ../models/
python test_PETA_hiarchical.py -g 0 -m hiarBayesGoogLeNet -c 61 -w ../models/
python test_PETA_hiarchical.py -g 1 -m hiarGoogLeNetWAM -c 61 -w ../models/
"""
from network.hiarGoogLenetSPP import hiarGoogLeNetSPP
from network.hiarGoogLenetWAM import hiarGoogLeNetWAM
from network.hiarGoogLenet import hiarGoogLeNet
from network.hiarGoogLenet_high import hiarGoogLeNet_high
from network.hiarGoogLenet_mid import hiarGoogLeNet_mid
from network.hiarGoogLenet_low import hiarGoogLeNet_low
from network.hiarBayesGoogLenet import hiarBayesGoogLeNet
from network.hiarBayesResNet import hiarBayesResNet
from network.hiarBayesInception_v4 import hiarBayesInception_v4
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
import re
import tqdm
from keras import backend as K
from angular_losses import weighted_categorical_crossentropy, coarse_to_fine_categorical_crossentropy_lowerbody

alpha = []

def parse_arg():
    models = ['hiarGoogLeNetSPP', 'hiarGoogLeNetWAM', 'hiarGoogLeNet', 'hiarBayesGoogLeNet']
    parser = argparse.ArgumentParser(description='training of the WPAL...')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='The gpu device\'s ID need to be used')
    parser.add_argument('-c', '--classes', type=int, default=65,
                        help='The total number of classes to be predicted')
    parser.add_argument('-b', '--batch', type=int, default=64,
                        help='The batch size of the training process')
    parser.add_argument('-wd', '--width', type=int, default=224,
                        help='The width of thWPAL_PETAe picture')
    parser.add_argument('-hg', '--height', type=int, default=224,
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
    save_name = ""
    low_level = [27, 32, 50, 56]#, 61, 62, 63, 64
    mid_level = [0, 6, 7, 8, 9, 11, 12, 13, 17, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 33, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 57, 58, 59, 60]
    high_level = [1, 2, 3, 4, 5, 10, 14, 15, 16, 18, 19, 31, 34, 40]
    """
    save_name = "binary61_cluster"
    low_level = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 33, 34, 35, 37, 38, 40, 41, 42, 43, 44, 46, 47, 48, 49, 54, 55, 56, 57, 59, 60]
    mid_level = [25, 32, 36, 39, 45]
    high_level = [0, 8, 18, 19, 50, 51, 52, 53, 58]
    """
    """
    save_name = "binary61_rl"
    low_level = [2, 4, 6, 8, 11, 13, 17, 19, 23, 24, 25, 32, 33, 34, 35, 37, 43]
    mid_level = [10, 15, 16, 20, 21, 26, 28, 29, 39, 40, 41, 42, 44, 48, 49, 50, 52, 53, 56, 60]
    high_level = [0, 1, 3, 5, 7, 9, 12, 14, 18, 22, 27, 30, 31, 36, 38, 45, 46, 47, 51, 54, 55, 57, 58, 59]
    """
    """
    low_level = [1, 2, 7, 12, 23, 27, 32, 34, 38, 40, 45, 46, 48, 55, 56, 58]
    mid_level = [0, 3, 4, 5, 6, 10, 11, 13, 14, 15, 16, 17, 21, 24, 25, 26, 28, 30, 31, 33, 35, 36, 37, 41, 42, 43, 44, 54, 57, 59, 60]
    high_level = [8, 18, 19, 51, 52, 53, 9, 20, 22, 29, 39, 47, 49, 50]
    """
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
            samplewise_std_normalization=False)
    else:
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False)
    image_width = args.width
    image_height = args.height
    filename = r"../results/PETA_sampled.csv"
    filename = r"/home/anhaoran/data/pedestrian_attributes_PETA/PETA/PETA.csv"
    data = np.array(pd.read_csv(filename))[:, 1:]
    length = len(data)
    data_x = np.zeros((length, image_width, image_height, 3))
    data_y = np.zeros((length, class_num))
    for i in range(7100, length):#11400
        #img = image.load_img(path + m)
        img = image.load_img(data[i, 0], target_size=(image_width, image_height, 3))
        data_x[i] = image.img_to_array(img)
        data_y[i] = np.array(data[i, 1:1+class_num], dtype="float32")
    data_y = data_y[:, list(np.hstack((low_level, mid_level, high_level)))]
    X_test = data_x[11400:]#7100
    y_test = data_y[11400:]#, len(low_level)+len(mid_level):#7100
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
        model_dir = "hiarGoogLeNetSPP_PETA/"
        model = hiarGoogLeNetSPP.build(None, None, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarGoogLeNetWAM":
        model_dir = "hiarGoogLeNetWAM_PETA/"
        model = hiarGoogLeNetWAM.build(None, None, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarGoogLeNet":
        model_dir = "hiarGoogLeNet_PETA/"
        model = hiarGoogLeNet.build(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarGoogLeNet_high":
        model_dir = "hiarGoogLeNet_PETA/"
        model = hiarGoogLeNet_high.build(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarGoogLeNet_mid":
        model_dir = "hiarGoogLeNet_PETA/"
        model = hiarGoogLeNet_mid.build(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarGoogLeNet_low":
        model_dir = "hiarGoogLeNet_PETA/"
        model = hiarGoogLeNet_low.build(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarBayesGoogLeNet":
        model_dir = "hiarBayesGoogLeNet_PETA/"
        #save_name = "binary61v2_multi"
        model = hiarBayesGoogLeNet.build(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarBayesInception_v4":
        model_dir = "hiarBayesInceptionV4_PETA/"
        model = hiarBayesInception_v4(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarBayesResNet":
        model_dir = "hiarBayesResNet_PETA/"
        model = hiarBayesResNet.build([len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics = ['accuracy']
    gpus_num = len(args.gpus.split(','))
    if gpus_num > 1:
        multi_gpu_model(model, gpus=gpus_num)
    #model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model.compile(loss=loss_func, optimizer='adam', loss_weights=loss_weights, metrics=metrics)
    model.summary()
    
    reg = args.weight + "_(e|f)1*"
    print(reg)
    weights = [s for s in os.listdir("../models/imagenet_models/" + model_dir) 
          if re.match(reg, s)]
    print(weights)
    for w in tqdm.tqdm(weights):
        model.load_weights("../models/imagenet_models/" + model_dir + w, by_name=True)
        predictions = model.predict(X_test)
        print("The shape of the predictions_test is: ", predictions.shape)
        np.save("../results/predictions/" + args.model + "_" + w + ".npy", predictions)
        print("../results/predictions/" + args.model + "_" + w + ".npy")