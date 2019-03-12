"""
python test_PETA_cluster.py -g 2 -c 61
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
from angular_losses import weighted_categorical_crossentropy, coarse_to_fine_categorical_crossentropy_lowerbody

alpha = []

def mA_acc(y_pred, y_true):
    M = len(y_pred)
    res = 0
    P = sum(y_true[:])
    N = M - P
    TP = sum(y_pred[:]*y_true[:])
    TN = list(y_pred[:]+y_true[:] == 0).count(True)
    if P != 0:
        res += TP/P + TN/N
    else:
        res += TN/N
    return res / 2

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
    attributes_list = ['accessoryHeadphone', 'personalLess15', 'personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'carryingBabyBuggy', 'carryingBackpack', 'hairBald', 'footwearBoots', 'lowerBodyCapri', 'carryingOther', 'carryingShoppingTro', 'carryingUmbrella', 'lowerBodyCasual', 'upperBodyCasual', 'personalFemale', 'carryingFolder', 'lowerBodyFormal', 'upperBodyFormal', 'accessoryHairBand', 'accessoryHat', 'lowerBodyHotPants', 'upperBodyJacket', 'lowerBodyJeans', 'accessoryKerchief', 'footwearLeatherShoes', 'upperBodyLogo', 'hairLong', 'lowerBodyLongSkirt', 'upperBodyLongSleeve', 'lowerBodyPlaid', 'lowerBodyThinStripes', 'carryingLuggageCase', 'personalMale', 'carryingMessengerBag', 'accessoryMuffler', 'accessoryNothing', 'carryingNothing', 'upperBodyNoSleeve', 'upperBodyPlaid', 'carryingPlasticBags', 'footwearSandals', 'footwearShoes', 'hairShort', 'lowerBodyShorts', 'upperBodyShortSleeve', 'lowerBodyShortSkirt', 'footwearSneakers', 'footwearStocking', 'upperBodyThinStripes', 'upperBodySuit', 'carryingSuitcase', 'lowerBodySuits', 'accessorySunglasses', 'upperBodySweater', 'upperBodyThickStripes', 'lowerBodyTrousers', 'upperBodyTshirt', 'upperBodyOther', 'upperBodyVNeck', 'footwear', 'hair', 'lowerbody', 'upperbody']


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
    X_test = data_x[11400:]
    y_test = data_y[11400:]
    print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    
    models = ["../results/models/model_low.hdf5", "../results/models/model_mid.hdf5", "../results/models/model_hig.hdf5"]
    model_names = ['low', 'mid', 'hig']
    loss_func = 'binary_crossentropy'
    loss_weights = None
    metrics=['accuracy']
    nb_epoch = 50
    batch_size = args.batch
    acc_features_for_all_attributes = []
    for j in range(class_num):
        acc_feature_for_one_attribute = []
        labels = y_test[:, j]
        for i in range(len(models)):
            model = load_model(models[i])
            gpus_num = len(args.gpus.split(','))
            if gpus_num != 1:
                multi_gpu_model(model, gpus=gpus_num)
            #model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
            model.compile(loss=loss_func, optimizer='adam', loss_weights=loss_weights, metrics=metrics)
            #model.summary()
            model.load_weights('../models/imagenet_models/GoogLeNet_PETA/iter50model_weightedloss_' + model_names[i] + '_attribute_' + str(j) + '_final_model.h5')
            predictions = model.predict(X_test)
            predictions = np.reshape(predictions, (-1,))
            predictions = np.array(predictions >= 0.5, dtype="float64")
            acc_feature_for_one_attribute.append(mA_acc(predictions, labels))
        acc_features_for_all_attributes.append(acc_feature_for_one_attribute)
        #print("-----------------------")
        print("[", str(j), "]", attributes_list[j], ": ", str(acc_feature_for_one_attribute))
        #print("-----------------------")
    acc_features_for_all_attributes = np.asarray(acc_features_for_all_attributes)
    np.save("../results/weighted_acc_features_for_all_attributes.npy", acc_features_for_all_attributes)