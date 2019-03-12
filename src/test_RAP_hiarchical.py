"""
python test_RAP_hiarchical.py -m hiarBayesGoogLeNet -b 64 -g 1 -w ../models/imagenet_models/hiarBayesGoogLeNet_RAP
python test_RAP_hiarchical.py -m hiarGoogLeNet -b 64 -g 1 -w ../models/imagenet_models/GoogLeNet_RAP
python test_RAP_hiarchical.py -m hiarBayesGoogLeNet -b 64 -g 1 -w ../models/imagenet_models/hiarBayesGoogLeNet_RAP/binary3_epoch50_valloss0.40.hdf5
"""
from network.hiarGoogLenetSPP import hiarGoogLeNetSPP
from network.hiarGoogLenetWAM import hiarGoogLeNetWAM
from network.hiarGoogLenet import hiarGoogLeNet
from network.hiarGoogLenet_high import hiarGoogLeNet_high
from network.hiarGoogLenet_mid import hiarGoogLeNet_mid
from network.hiarGoogLenet_low import hiarGoogLeNet_low
from network.hiarBayesGoogLenet import hiarBayesGoogLeNet
from network.hiarBayesInception_v4 import hiarBayesInception_v4
from network.hiarBayesGoogLenet_inception import hiarBayesGoogLeNet as hiarBayesGoogLeNet_inception
from network.hiarGoogLenet_inception import hiarGoogLeNet as hiarGoogLeNet_inception
from network.hiarBayesGoogLenetv2 import hiarBayesGoogLeNet as hiarBayesGoogLeNetv2
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
import tqdm
from keras import backend as K
from angular_losses import bayes_binary_crossentropy
import re

alpha = []

def parse_arg():
    models = ['hiarGoogLeNet', 'hiarBayesGoogLeNet', 'hiarBayesGoogLeNet_inception']
    parser = argparse.ArgumentParser(description='training of the WPAL...')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='The gpu device\'s ID need to be used')
    parser.add_argument('-c', '--classes', type=int, default=92,
                        help='The total number of classes to be predicted')
    parser.add_argument('-b', '--batch', type=int, default=64,
                        help='The batch size of the training process')
    parser.add_argument('-wd', '--width', type=int, default=120,
                        help='The width of thWPAL_PETAe picture')
    parser.add_argument('-hg', '--height', type=int, default=320,
                        help='The height of the picture')
    parser.add_argument('-w', '--weight', type=str, default='',
                        help='The weights file of the pre-training')
    parser.add_argument('-m', '--model', type=str, default='',
                        help='The model including: '+str(models))
    parser.add_argument('-s', '--split', type=int, default=0,
                        help='The split')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args


    


if __name__ == "__main__":
    args = parse_arg()
    #"""
    save_name = "binary51_75v2_"
    low_level = [11]#,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91
    mid_level = [9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
    high_level = [0,1,2,3,4,5,6,7,8,43,44,45,46,47,48,49,50]#,51,52,53,54,55,56,57,58,59,60,61,62
    #"""
    save_name = "binary92_oldhiar_newlossnoexp_split" + str(args.split)
    low_level = [11,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91]#
    mid_level = [9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
    high_level = [0,1,2,3,4,5,6,7,8,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62]#
    #"""
    save_name = "binary51_newhier"
    low_level = [11]#,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91
    mid_level = [4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    high_level = [0,1,2,3]#,51,52,53,54,55,56,57,58,59,60,61,62
    #"""
    class_num = args.classes


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
    filename = r"../results/RAP_labels_pd.csv"
    data = np.array(pd.read_csv(filename))[:, 1:]
    length = len(data)
    #global alpha
    data_x = np.zeros((length, image_height, image_width, 3))
    data_y = np.zeros((length, class_num))
    for i in range(length):
        #img = image.load_img(path + m)
        img = image.load_img(data[i, 0], target_size=(image_height, image_width, 3))
        data_x[i] = image.img_to_array(img)
        data_y[i] = np.array(data[i, 1:1+class_num], dtype="float32")
    data_y = data_y[:, list(np.hstack((low_level, mid_level, high_level)))]
    #X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)
    split = np.load('../results/RAP_partion.npy').item()
    X_test = data_x[list(split['test'][args.split])]
    y_test = data_y[list(split['test'][args.split])]
    print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    
    
    #googleNet默认输入32*32的图片
    if args.model == "hiarGoogLeNet":
        model_dir = "hiarGoogLeNet_RAP/"
        model = hiarGoogLeNet.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarBayesGoogLeNet":
        model_dir = "hiarBayesGoogLeNet_RAP/"
        model = hiarBayesGoogLeNet.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarBayesGoogLeNet_inception":
        model_dir = "hiarBayesGoogLeNet_Inception_RAP/"
        model = hiarBayesGoogLeNet_inception.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarGoogLeNet_inception":
        model_dir = "hiarGoogLeNet_Inception_RAP/"
        model = hiarGoogLeNet_inception.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarBayesGoogLeNetv2":
        model = hiarBayesGoogLeNetv2.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarBayesInception_v4":
        model_dir = "hiarBayesInceptionV4_RAP/"
        model = hiarBayesInception_v4(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
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
        predictions_list = model.predict(X_test)
        if args.model == "hiarBayesGoogLeNetv2":
            predictions = np.array(predictions_list).reshape((class_num, -1)).T
        else:
            predictions = np.array(predictions_list)
        print("The shape of the predictions_test is: ", predictions.shape)
        #w[w.rindex('/')+1:w.rindex('.')]
        np.save("../results/predictions/" + args.model + '_' + save_name + w + "_predictions_imagenet_test_RAP.npy", predictions)
        print("../results/predictions/" + args.model + '_' + save_name + w + "_predictions_imagenet_test_RAP.npy")