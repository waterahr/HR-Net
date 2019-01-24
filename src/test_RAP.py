"""
python train_RAP_hiarchical.py -m hiarBayesGoogLeNet -b 64 -g 1 -w ../models/imagenet_models/hiarBayesGoogLeNet_RAP
python test_RAP.py -m GoogLeNet -w ../models/imagenet_models/GoogLeNet_RAP/binary51_b2_lr0.0002_lossweight_final_model.h5 -c 51
"""
from network.GoogleLenet import GoogLeNet
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
    models = ['GoogLeNet']
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
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args


    


if __name__ == "__main__":
    #"""
    save_name = "binary51_"
    #save_name = "binary3_"
    #part = [2,11,24]
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
    #hiarBayesGoogLeNet
    if args.model == "GoogLeNet":
        #filename = r"../results/RAP_labels_pd.csv"
        filename = r"../results/myRAP_labels_pd.csv"
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
    #data_y = data_y[:, part]
    #class_num = 3
    #X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)
    """
    split = np.load('../results/RAP_partion.npy').item()
    X_test = data_x[list(split['test'][0])]
    y_test = data_y[list(split['test'][0])]
    """
    X_test = data_x[33268:]
    y_test = data_y[33268:]
    print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    
    
    #googleNet默认输入32*32的图片
    if args.model == "GoogLeNet":
        model = GoogLeNet.build(None, None, 3, class_num)
        loss_func = 'binary_crossentropy'
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
    np.save("../results/predictions/" + args.model+ '_' + save_name + args.weight[args.weight.rindex('/')+1:-3] + "_predictions_imagenet_test_RAP.npy", predictions)
