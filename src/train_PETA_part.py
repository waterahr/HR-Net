"""
python train_PETA_part.py -m partGoogLeNet -c 61 -g 1 -w 
"""
from network.partGoogLeNet import partGoogLeNet
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
    models = ['partGoogLeNet']
    parser = argparse.ArgumentParser(description='training of the WPAL...')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='The gpu device\'s ID need to be used')
    parser.add_argument('-c', '--classes', type=int, default=65,
                        help='The total number of classes to be predicted')
    parser.add_argument('-b', '--batch', type=int, default=256,
                        help='The batch size of the training process')
    parser.add_argument('-wd', '--width', type=int, default=160,
                        help='The width of thWPAL_PETAe picture')
    parser.add_argument('-hg', '--height', type=int, default=75,
                        help='The height of the picture')
    parser.add_argument('-w', '--weight', type=str, default='',
                        help='The weights file of the pre-training')
    parser.add_argument('-m', '--model', type=str, default='partGoogLeNet',
                        help='The model including: '+str(models))
    parser.add_argument('-i', '--iteration', type=int, default=50,
                        help='The model iterations')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args


    


if __name__ == "__main__":
    #"""
    save_name = "binary61_"
    head_level = [0,8,20,21,25,28,36,37,44,54]
    upperbody_level = [7,15,17,19,23,27,30,39,40,46,50,51,55,56,58,59,60]
    lowerbody_level = [6,10,11,12,13,14,18,22,24,29,31,32,33,35,38,41,45,47,52,53,57]
    foot_level = [9,26,42,43,48,49]
    global_level = [1,2,3,4,5,16,34]
    ###rl
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
    #partGoogLeNet
    if args.model == "partGoogLeNet":
        filename = r"../results/PETA.csv"
    data = np.array(pd.read_csv(filename))[:, 1:]
    length = len(data)
    #global alpha
    data_x = np.zeros((length, image_width, image_height, 3))
    """
    data_y_low = np.zeros((length, len(low_level)))
    data_y_mid = np.zeros((length, len(mid_level)))
    data_y_hig = np.zeros((length, len(high_level)))
    for i in range(length):
        #img = image.load_img(path + m)
        img = image.load_img(data[i, 0], target_size=(image_width, image_height, 3))
        data_x[i] = image.img_to_array(img)
        data_y_low[i] = np.array(data[i, low_level], dtype="float32")
        data_y_mid[i] = np.array(data[i, mid_level], dtype="float32")
        data_y_hig[i] = np.array(data[i, high_level], dtype="float32")
    data_y = np.hstack((data_y_low, data_y_mid, data_y_hig))
    """
    data_y = np.zeros((length, class_num))
    for i in range(length):
        #img = image.load_img(path + m)
        img = image.load_img(data[i, 0], target_size=(image_width, image_height, 3))
        data_x[i] = image.img_to_array(img)
        data_y[i] = np.array(data[i, 1:1+class_num], dtype="float32")
    data_y = data_y[:, list(np.hstack((head_level, upperbody_level, lowerbody_level, foot_level, global_level)))]
    #X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)
    X_train = data_x[:9500]
    X_test = data_x[9500:11400]
    y_train = data_y[:9500]#, len(low_level)+len(mid_level):
    y_test = data_y[9500:11400]#, len(low_level)+len(mid_level):
    print("The shape of the X_train is: ", X_train.shape)
    print("The shape of the y_train is: ", y_train.shape)
    print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    #np.save("../results/" + args.model + '_' + save_name + "_X_test.npy", X_test)
    #np.save("../results/" + args.model + '_' + save_name + "_y_test.npy", y_test)
    
    
    #googleNet默认输入32*32的图片
    if args.model == "partGoogLeNet":
        model = partGoogLeNet.build(None, None, 3, [head_level, upperbody_level, lowerbody_level, foot_level, global_level])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
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
    if args.model == "partGoogLeNet":
        model_dir = 'partGoogLeNet_PETA'
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
    #print(train_generator)
    """
            initial_epoch = 100,
    """
    model.fit_generator(train_generator,
            steps_per_epoch = int(X_train.shape[0] / (batch_size * gpus_num)),
            epochs = nb_epoch,
            validation_data = val_generator,
            validation_steps = int(X_test.shape[0] / (batch_size * gpus_num)),
            callbacks = [checkpointer, csvlog],
            initial_epoch = 50)
    model.save_weights('../models/imagenet_models/' + model_dir + '/' + save_name+ '_final'+str(args.iteration)+'iter_model.h5')