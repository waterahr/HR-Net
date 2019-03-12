"""
python train_PETA.py -g 0,1 -c 61 -b 256 -w ../models/xxxxx.hdf5 -m GoogLeNetSPP
python train_PETA.py -g 0,1 -c 68 -b 256 -w ../models/xxxxx.hdf5 -m OEDCGoogLeNetSPP
python train_PETA.py -g 0 -c 14 -b 64 -m OEDCGoogLeNetSPP_lowerBody  -w ../models/xxxxx.hdf5
"""
from network.GoogLenetSPP import GoogLeNetSPP
from network.GoogleLenet import GoogLeNet
from network.GoogleLenet_gap import GoogLeNet as GoogLeNet_gap
from network.OEDC_GoogLenetSPP import OEDCGoogLeNetSPP
from network.OEDC_GoogLenetSPP_lowerBody import OEDCGoogLeNetSPP_lowerBody
from network.Inception_v4 import Inception_v4
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
from angular_losses import weighted_binary_crossentropy

def weighted_acc(y_true, y_pred):
    return K.mean(K.mean(K.equal(y_true, K.round(y_pred)), axis=-1), axis=-1)

alpha = []

def generate_imgdata_from_file(X_path, y, batch_size, image_height, image_width):
    while True:
        cnt = 0
        X = []
        Y = []
        for i in range(len(X_path)):
            img = image.load_img(X_path[i], target_size=(image_height, image_width, 3))
            img = image.img_to_array(img)
            X.append(img)
            Y.append(y[i])
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                yield (np.array(X), np.array(Y))
                X = []
                Y = []

def parse_arg():
    models = ['GoogLeNet', 'GoogLeNet_gap', 'Inception_v4', 'GoogLeNetSPP', 'OEDCGoogLeNetSPP', 'OEDCGoogLeNetSPP_lowerBody']
    parser = argparse.ArgumentParser(description='training of the WPAL...')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='The gpu device\'s ID need to be used')
    parser.add_argument('-c', '--classes', type=int, default=65,
                        help='The total number of classes to be predicted')
    parser.add_argument('-b', '--batch', type=int, default=64,
                        help='The batch size of the training process')
    parser.add_argument('-wd', '--width', type=int, default=160,
                        help='The width of thWPAL_PETAe picture')
    parser.add_argument('-hg', '--height', type=int, default=75,
                        help='The height of the picture')
    parser.add_argument('-w', '--weight', type=str, default='',
                        help='The weights file of the pre-training')
    parser.add_argument('-m', '--model', type=str, default='',
                        help='The model including: '+str(models))
    parser.add_argument('-d', '--depth', type=int, default=9,
                        help='The model depth')
    parser.add_argument('-i', '--iteration', type=int, default=50,
                        help='The model iterations')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args


    


if __name__ == "__main__":
    args = parse_arg()
    #save_name = "binary61_depth" + str(args.depth)
    save_name = "binary61_newlossnoexp"
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
        filename = r"../results/PETA.csv"
    elif args.model == "GoogLeNet" or args.model == "GoogLeNet_gap" or args.model == 'Inception_v4':
        filename = r"../results/PETA.csv"
    elif args.model == "OEDCGoogLeNetSPP":
        filename = r"../results/PETA_coarse_to_fine_labels_pd.csv"
    elif args.model == "OEDCGoogLeNetSPP_lowerBody":
        filename = r"../results/PETA_lowerBody_labels_pd.csv"
    data = np.array(pd.read_csv(filename))[:, 1:]
    length = len(data)
    #global alpha
    data_x = np.zeros((length, image_width, image_height, 3))
    data_y = np.zeros((length, class_num))
    data_path = []
    load = False
    for i in range(length):
        #img = image.load_img(path + m)
        data_path.append(data[i, 0])
        if load:
            img = image.load_img(data[i, 0], target_size=(image_height, image_width, 3))
            data_x[i] = image.img_to_array(img)
        data_y[i] = np.array(data[i, 1:1+class_num], dtype="float32")
    data_path = np.array(data_path)
    for i in range(class_num):
        alpha[i] += list(data_y[:, i]).count(1.0)
    alpha /= length
    print("The positive ratio of each attribute is:\n", alpha)
    #X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)
    if load:
        X_train = data_x[:9500]
        X_test = data_x[9500:11400]
    X_train_path = data_path[:9500]
    X_test_path = data_path[9500:11400]
    y_train = data_y[:9500]
    y_test = data_y[9500:11400]
    if load:
        print("The shape of the X_train is: ", X_train.shape)
    print("The shape of the y_train is: ", y_train.shape)
    if load:
        print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    #np.save("../results/" + args.model + '_' + save_name + "_X_test.npy", X_test)
    #np.save("../results/" + args.model + '_' + save_name + "_y_test.npy", y_test)
    
    
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
    elif args.model == "Inception_v4":
        model = Inception_v4(image_width, image_height, 3, class_num)
        loss_func = 'binary_crossentropy'
        loss_func = weighted_binary_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
        metrics = [weighted_acc]
    elif args.model == "GoogLeNet_gap":
        model = GoogLeNet_gap.build(image_width, image_height, 3, class_num, model_depth=args.depth)
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


    nb_epoch = args.iteration
    batch_size = args.batch
    if load:
        train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    else:
        train_generator = generate_imgdata_from_file(X_train_path, y_train, batch_size, image_height, image_width)
    if load:
        val_generator = datagen.flow(X_test, y_test, batch_size=batch_size)
    else:
        val_generator = generate_imgdata_from_file(X_test_path, y_test, batch_size, image_height, image_width)
    monitor = 'val_loss'
    if args.model == "GoogLeNetSPP":
        model_dir = 'GoogLeNetSPP_PETA'
    elif args.model == "GoogLeNet":
        model_dir = 'GoogLeNet_PETA'
    elif args.model == "Inception_v4":
        model_dir = "InceptionV4_PETA"
    elif args.model == "GoogLeNet_gap":
        model_dir = 'GoogLeNetGAP_PETA'
    elif args.model == "OEDCGoogLeNetSPP":
        model_dir = 'OEDCWPAL_PETA'
    elif args.model == "OEDCGoogLeNetSPP_lowerBody":
        model_dir = 'OEDCWPAL_PETA_lowerBody'
    checkpointer = ModelCheckpoint(filepath = '../models/imagenet_models/' + model_dir + '/' + save_name+ '_epoch{epoch:02d}_valloss{'+ monitor + ':.2f}.hdf5',
                                   monitor = monitor,
                                   verbose=1, 
                                   save_best_only=True, 
                                   save_weights_only=True,
                                   mode='auto', 
                                   period=50)
    csvlog = CSVLogger('../models/imagenet_models/' + model_dir + '/' + save_name+'_'+str(args.iteration)+'iter'+'_log.csv')#, append=True
    if args.weight != '':
        model.load_weights(args.weight, by_name=True)
    model.fit_generator(train_generator,
            steps_per_epoch = int(X_train_path.shape[0] / (batch_size * gpus_num)),
            epochs = nb_epoch,
            validation_data = val_generator,
            validation_steps = int(X_test_path.shape[0] / (batch_size * gpus_num)),
            callbacks = [checkpointer, csvlog])
    if args.model == "GoogLeNetSPP":
        model.save_weights('../models/imagenet_models/' + model_dir + '/' + save_name+ '_final'+str(args.iteration)+'iter_model.h5')
    elif args.model == "GoogLeNet" or args.model == "GoogLeNet_gap" or args.model == "Inception_v4":
        model.save_weights('../models/imagenet_models/' + model_dir + '/' + save_name+ '_final'+str(args.iteration)+'iter_model.h5')
    elif args.model == "OEDCGoogLeNetSPP":
        model.save_weights('../models/imagenet_models/' + model_dir + '/' + save_name+ '_final'+str(args.iteration)+'iter_model.h5')
    elif args.model == "OEDCGoogLeNetSPP_lowerBody":
        model.save_weights('../models/imagenet_models/' + model_dir + '/' + save_name+ '_train_final'+str(args.iteration)+'iter_model.h5')
        #model_pred.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        #model_pred.load_weights('../models/imagenet_models/' + model_dir + '/train_final_model.h5', by_name=True)
        #model_pred.save('../models/imagenet_models/' + model_dir + '/final_model.h5')
