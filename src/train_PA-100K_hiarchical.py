"""
python train_PA-100K_hiarchical.py -m hiarBayesGoogLeNet -b 64 -g 1 -w ../models/imagenet_models/hiarBayesGoogLeNet_PA-100K
python train_PA-100K_hiarchical.py -m hiarGoogLeNet -b 64 -g 1 -w ../models/imagenet_models/hiarGoogLeNet_PA-100K
python train_PA-100K_hiarchical.py -m hiarBayesGoogLeNet -b 64 -g 1 -c 26 -w ../models/imagenet_models/hiarBayesGoogLeNet_PA-100K
python train_PA-100K_hiarchical.py -m hiarGoogLeNet -b 64 -g 1 -c 26 -w ../models/imagenet_models/hiarGoogLeNet_PA-100K
"""
from network.hiarGoogLenetSPP import hiarGoogLeNetSPP
from network.hiarGoogLenetWAM import hiarGoogLeNetWAM
from network.hiarGoogLenet import hiarGoogLeNet
from network.hiarGoogLenet_high import hiarGoogLeNet_high
from network.hiarGoogLenet_mid import hiarGoogLeNet_mid
from network.hiarGoogLenet_low import hiarGoogLeNet_low
from network.hiarBayesGoogLenet import hiarBayesGoogLeNet
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
from keras import backend as K
from angular_losses import bayes_binary_crossentropy

def multi_generator(generator):
    while True:
        x, y = generator.next()
        y_list = []
        for i in range(y.shape[1]):
            y_list.append(y[:, i])
        yield x, y_list

alpha = []

def parse_arg():
    models = ['hiarGoogLeNet', 'hiarBayesGoogLeNet']
    parser = argparse.ArgumentParser(description='training of the WPAL...')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='The gpu device\'s ID need to be used')
    parser.add_argument('-c', '--classes', type=int, default=26,
                        help='The total number of classes to be predicted')
    parser.add_argument('-b', '--batch', type=int, default=64,
                        help='The batch size of the training process')
    parser.add_argument('-wd', '--width', type=int, default=75,
                        help='The width of thWPAL_PETAe picture')
    parser.add_argument('-hg', '--height', type=int, default=160,
                        help='The height of the picture')
    parser.add_argument('-w', '--weight', type=str, default='',
                        help='The weights file of the pre-training')
    parser.add_argument('-m', '--model', type=str, default='',
                        help='The model including: '+str(models))
    parser.add_argument('-i', '--iteration', type=int, default=50,
                        help='The model iterations')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args


    


if __name__ == "__main__":
    save_name = "binary26_"
    low_level = [15,16,17,18,19,20]
    mid_level = [7,8,9,10,11,12,13,14,21,22,23,24,25]
    high_level = [0,1,2,3,4,5,6]
    """
    save_name = "binary3"
    low_level = [11]
    mid_level = [24]
    high_level = [2]
    """
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
    if args.model == "hiarGoogLeNet":
        filename = r"../results/PA-100K_labels_pd.csv"
    elif args.model == "hiarBayesGoogLeNet":
        filename = r"../results/PA-100K_labels_pd.csv"
    elif args.model == "hiarBayesGoogLeNetv2":
        filename = r"../results/PA-100K_labels_pd.csv"
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
    X_train = data_x[:80000]
    X_test = data_x[80000:90000]
    y_train = data_y[:80000]#, len(low_level)+len(mid_level):
    y_test = data_y[80000:90000]#, len(low_level)+len(mid_level):
    print("The shape of the X_train is: ", X_train.shape)
    print("The shape of the y_train is: ", y_train.shape)
    print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    
    
    #googleNet默认输入32*32的图片
    if args.model == "hiarGoogLeNet":
        model = hiarGoogLeNet.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarBayesGoogLeNet":
        model = hiarBayesGoogLeNet.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarBayesGoogLeNetv2":
        model = hiarBayesGoogLeNetv2.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    gpus_num = len(args.gpus.split(','))
    if gpus_num > 1:
        multi_gpu_model(model, gpus=gpus_num)
    #model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model.compile(loss=loss_func, optimizer='adam', loss_weights=loss_weights, metrics=metrics)
    model.summary()


    nb_epoch = args.iteration
    batch_size = args.batch
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = datagen.flow(X_test, y_test, batch_size=batch_size)
    monitor = 'val_loss'
    if args.model == "hiarGoogLeNet":
        model_dir = 'hiarGoogLeNet_PA-100K'
    elif args.model == "hiarBayesGoogLeNet" or args.model == "hiarBayesGoogLeNetv2":
        model_dir = 'hiarBayesGoogLeNet_PA-100K'
    checkpointer = ModelCheckpoint(filepath = '../models/imagenet_models/' + model_dir + '/' + save_name+ '_epoch{epoch:02d}_valloss{'+ monitor + ':.2f}.hdf5',
                                   monitor = monitor,
                                   verbose=1, 
                                   save_best_only=True, 
                                   save_weights_only=True,
                                   mode='auto', 
                                   period=25)
    csvlog = CSVLogger('../models/imagenet_models/' + model_dir + '/' + save_name+'_'+str(args.iteration)+'iter'+'_log.csv', append=True)#
    if args.weight != '':
        model.load_weights(args.weight, by_name=True)
    print(train_generator)
    """
            , initial_epoch = 50,
    """
    #multi_generator(train_generator),multi_generator(val_generator)
    model.fit_generator(multi_generator(train_generator),
            steps_per_epoch = int(X_train.shape[0] / (batch_size * gpus_num)),
            epochs = nb_epoch,
            validation_data = multi_generator(val_generator),
            validation_steps = int(X_test.shape[0] / (batch_size * gpus_num)),
            callbacks = [checkpointer, csvlog])
    model.save_weights('../models/imagenet_models/' + model_dir + '/' + save_name+ '_final'+str(args.iteration)+'iter_model.h5')
