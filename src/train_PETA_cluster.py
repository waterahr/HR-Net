"""
python train_PETA_cluster.py -g 2 -c 61 -b 512
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from keras.models import load_model
import sys
import os
import argparse
import json
import h5py
import numpy as np
import pandas as pd
from keras import backend as K
from angular_losses import weighted_binary_crossentropy

alpha = []

def parse_arg():
    parser = argparse.ArgumentParser(description='training of the WPAL...')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='The gpu device\'s ID need to be used')
    parser.add_argument('-b', '--batch', type=int, default=64,
                        help='The batch size of the training process')
    parser.add_argument('-c', '--classes', type=int, default=65,
                        help='The total number of classes to be predicted')
    parser.add_argument('-wd', '--width', type=int, default=160,
                        help='The width of thWPAL_PETAe picture')
    parser.add_argument('-hg', '--height', type=int, default=75,
                        help='The height of the picture')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args
    

if __name__ == "__main__":
    args = parse_arg()
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
    filename = r"../results/PETA.csv"
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
    X_train = data_x[:9500]
    X_test = data_x[9500:11400]
    y_train = data_y[:9500]
    y_test = data_y[9500:11400]
    print("The shape of the X_train is: ", X_train.shape)
    print("The shape of the y_train is: ", y_train.shape)
    print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    alpha = np.sum(data_y[:11400], axis=0)#(len(data_y,), )
    alpha /= len(data_y[:11400])
    print(alpha)
    
    models = []
    model_names = ['low', 'mid', 'hig']
    models.append("../results/models/model_low.hdf5")
    models.append("../results/models/model_mid.hdf5")
    models.append("../results/models/model_hig.hdf5")
    loss_func = 'binary_crossentropy'
    loss_weights = None
    metrics=['accuracy']
    nb_epoch = 50
    batch_size = args.batch
    for i in range(len(models)):
        for j in range(class_num):
            print("-----------------------")
            print(models[i], ":", str(j))
            print("-----------------------")
            model = load_model(models[i])
            gpus_num = len(args.gpus.split(','))
            if gpus_num != 1:
                multi_gpu_model(model, gpus=gpus_num)
            #model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
            loss_func = weighted_binary_crossentropy(alpha[j])
            loss_weights = None
            #metrics = [weighted_acc]
            model.compile(loss=loss_func, optimizer='adam', loss_weights=loss_weights, metrics=metrics)
            model.summary()

            train_generator = datagen.flow(X_train, y_train[:, j], batch_size=batch_size)
            val_generator = datagen.flow(X_test, y_test[:, j], batch_size=batch_size)
            model.fit_generator(train_generator,
                    steps_per_epoch = int(X_train.shape[0] / (batch_size * gpus_num)),
                    epochs = nb_epoch,
                    validation_data = val_generator,
                    validation_steps = int(X_test.shape[0] / (batch_size * gpus_num)),
                    callbacks = [ModelCheckpoint('../models/imagenet_models/GoogLeNet_PETA/iter50model_weightedloss_' + model_names[i] + '_attribute_' + str(j) + '_best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)])#EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
            model.save_weights('../models/imagenet_models/GoogLeNet_PETA/iter50model_weightedloss_' + model_names[i] + '_attribute_' + str(j) + '_final_model.h5')