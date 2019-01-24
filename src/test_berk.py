"""
python test_berk.py -g 1 -c 9 -b 32 -m GoogLeNet -w ../models/imagenet_models/GoogLeNet_berk/binary9-depth9_epoch25_valloss0.64.hdf5
"""
from network.GoogLenetSPP import GoogLeNetSPP
from network.GoogleLenet import GoogLeNet
from network.OEDC_GoogLenetSPP import OEDCGoogLeNetSPP
from network.OEDC_GoogLenetSPP_lowerBody import OEDCGoogLeNetSPP_lowerBody
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
import sys
import os
import argparse
import json
import h5py
import numpy as np
import pandas as pd
from keras import backend as K
from angular_losses import weighted_categorical_crossentropy, coarse_to_fine_categorical_crossentropy_lowerbody

alpha = []

def parse_arg():
    models = ['GoogLeNet', 'GoogLeNetSPP', 'OEDCGoogLeNetSPP', 'OEDCGoogLeNetSPP_lowerBody']
    parser = argparse.ArgumentParser(description='training of the WPAL...')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='The gpu device\'s ID need to be used')
    parser.add_argument('-c', '--classes', type=int, default=9,
                        help='The total number of classes to be predicted')
    parser.add_argument('-b', '--batch', type=int, default=32,
                        help='The batch size of the training process')
    parser.add_argument('-wd', '--width', type=int, default=420,
                        help='The width of thWPAL_berke picture')
    parser.add_argument('-hg', '--height', type=int, default=210,
                        help='The height of the picture')
    parser.add_argument('-w', '--weight', type=str, default='',
                        help='The weights file of the pre-training')
    parser.add_argument('-m', '--model', type=str, default='',
                        help='The model including: '+str(models))
    parser.add_argument('-d', '--depth', type=int, default=9,
                        help='The model depth')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args


    


if __name__ == "__main__":
    args = parse_arg()
    save_name = "binary9-depth" + str(args.depth)
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
    if args.model == "GoogLeNetSPP":
        filename = r"../results/berk-test.csv"
    elif args.model == "GoogLeNet":
        filename = r"../results/berk-test.csv"
    elif args.model == "OEDCGoogLeNetSPP":
        filename = r"../results/berk_coarse_to_fine_labels_pd.csv"
    elif args.model == "OEDCGoogLeNetSPP_lowerBody":
        filename = r"../results/berk_lowerBody_labels_pd.csv"
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
    X_test = data_x#[11400:]
    y_test = data_y#[11400:]
    print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    
    
    #googleNet默认输入32*32的图片
    if args.model == "GoogLeNetSPP":
        model = GoogLeNetSPP.build(None, None, 3, class_num)
        loss_func = 'binary_crossentropy'
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "GoogLeNet":
        model = GoogLeNet.build(image_width, image_height, 3, class_num, model_depth=args.depth)
        loss_func = 'binary_crossentropy'
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "OEDCGoogLeNetSPP":
        model = OEDCGoogLeNetSPP.build(None, None, 3, 7, [3, 7, 11, 6, 7, 12, 15])#[4, 7, 11, 7, 7, 13, 16]
        loss_func = 'binary_crossentropy'
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "OEDCGoogLeNetSPP_lowerBody":
        model = OEDCGoogLeNetSPP_lowerBody.build(None, None, 3, 2, 7, 5)
        loss_func = 'binary_crossentropy'#coarse_to_fine_categorical_crossentropy_lowerbody(alpha)#['categorical_crossentropy', lambda y_true,y_pred: y_pred]
        loss_weights=None#[1.,1.]
        metrics={'softmax_labels':'accuracy'}
    gpus_num = len(args.gpus.split(','))
    if gpus_num != 1:
        multi_gpu_model(model, gpus=gpus_num)
    #model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model.compile(loss=loss_func, optimizer='adam', loss_weights=loss_weights, metrics=metrics)
    model.summary()


    model.load_weights(args.weight, by_name=True)
    
    predictions = model.predict(X_test)
    print("The shape of the predictions_test is: ", predictions.shape)
    np.save("../results/predictions/" + args.model+ '_' + save_name + "_predictions_imagenet_berk.npy", predictions)
