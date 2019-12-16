"""
python train_PETA.py -g 0,1 -c 61 -b 64 -m hiarGoogLeNetSPP
python train_PETA_hiarchical.py -m hiarGoogLeNet -c 61 -g 1
python train_PETA_hiarchical.py -m hiarBayesGoogLeNet -c 61 -g 1 -w ../models/imagenet_models/hiarBayesGoogLeNet_PETA/binary61_multi_final_model.h5/inary61_multi_mar_final_model.h5
python train_PETA_hiarchical.py -m hiarGoogLeNetWAM -c 61 -g 1
python train_PETA_hiarchical.py -m hiarBayesInception_v4 -c 61 -i 200 -wd 299 -hg 299 -b 32 -g 
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
from keras.optimizers import *
import os
import argparse
import json
import numpy as np
import pandas as pd
from keras import backend as K
from angular_losses import bayes_binary_crossentropy
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
                
def mA(y_true, y_pred):
    """
    y_pred_np = K.eval(y_pred)
    y_true_np = K.eval(y_true)
    M = len(y_pred_np)
    L = len(y_pred_np[0])
    res = 0
    for i in range(L):
        P = sum(y_true_np[:, i])
        N = M - P
        TP = sum(y_pred_np[:, i]*y_true_np[:, i])
        TN = list(y_pred_np[:, i]+y_true_np[:, i] == 0.).count(True)
        #print(TP, P, TN, N)
        #print(P,',', N,',', TP,',', TN)
        #if P != 0:
        res += TP/P + TN/N
    return res / (2*L)
    
    y_pred = K.cast(y_pred >= 0.5, dtype='float32')
    y_true = K.cast(y_true >= 0.5, dtype='float32')
    #print(K.int_shape(y_true))
    P = K.sum(y_true, axis=-1) + K.epsilon()
    #print("P", P)
    N = K.sum(1-y_true, axis=-1) + K.epsilon()
    #print("N", N)
    TP = K.sum(y_pred * y_true, axis=-1)
    #print("TP", TP)
    TN = K.sum(K.cast_to_floatx(y_pred + y_true == 0))
    #print("TN", TN)
    return K.mean(TP / P + TN / N) / 2"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=-1)
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)), axis=-1)
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)), axis=-1)
    mean_acc = (true_positives / (possible_positives + K.epsilon()) + true_negatives / (possible_negatives + K.epsilon())) / 2
    return mean_acc

def parse_arg():
    models = ['hiarGoogLeNetSPP', 'hiarGoogLeNetWAM', 'hiarGoogLeNet', 'hiarBayesGoogLeNet', 'hiarGoogLeNet_high', 'hiarGoogLeNet_mid', 'hiarGoogLeNet_low', 'hiarBayesResNet']
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
    parser.add_argument('-i', '--iteration', type=int, default=50,
                        help='The model iterations')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args


    


if __name__ == "__main__":
    #"""
    #save_name = "binary61_newlossnoexp_sampled"
    save_name = "sgd"
    save_name = "adam"
    low_level = [27, 32, 50, 56]#, 61, 62, 63, 64
    mid_level = [0, 6, 7, 8, 9, 11, 12, 13, 17, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 33, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 57, 58, 59, 60]
    high_level = [1, 2, 3, 4, 5, 10, 14, 15, 16, 18, 19, 31, 34, 40]
    #"""
    """
    save_name = "binary61_cluster"
    low_level = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 33, 34, 35, 37, 38, 40, 41, 42, 43, 44, 46, 47, 48, 49, 54, 55, 56, 57, 59, 60]
    mid_level = [25, 32, 36, 39, 45]
    high_level = [0, 8, 18, 19, 50, 51, 52, 53, 58]
    """
    ###rl
    """
    save_name = "rl_binary61"
    low_level = [2, 4, 6, 8, 11, 13, 17, 19, 23, 24, 25, 32, 33, 34, 35, 37, 43]
    mid_level = [10, 15, 16, 20, 21, 26, 28, 29, 39, 40, 41, 42, 44, 48, 49, 50, 52, 53, 56, 60]
    high_level = [0, 1, 3, 5, 7, 9, 12, 14, 18, 22, 27, 30, 31, 36, 38, 45, 46, 47, 51, 54, 55, 57, 58, 59]
    """
    """
    low_level = [27, 31, 32, 40, 50, 56, 60]#, 61, 62, 63, 64
    mid_level = [0, 6, 7, 8, 9, 11, 12, 13, 17, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 33, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 57, 58, 59]
    high_level = [1, 2, 3, 4, 5, 10, 14, 15, 16, 18, 19, 34]
    """
    """
    low_level = [1, 2, 7, 12, 23, 27, 32, 34, 38, 40, 45, 46, 48, 55, 56, 58]
    mid_level = [0, 3, 4, 5, 6, 10, 11, 13, 14, 15, 16, 17, 21, 24, 25, 26, 28, 30, 31, 33, 35, 36, 37, 41, 42, 43, 44, 54, 57, 59, 60]
    high_level = [8, 18, 19, 51, 52, 53, 9, 20, 22, 29, 39, 47, 49, 50]
    """
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
        datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True)
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
    filename = r"../results/PETA_sampled.csv"
    filename = r"/home/anhaoran/data/pedestrian_attributes_PETA/PETA/PETA.csv"
    data = np.array(pd.read_csv(filename))[:, 1:]
    length = len(data)
    print(length)
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
    data_path = []
    data_y = np.zeros((length, class_num))
    load = False
    for i in range(length):
        #img = image.load_img(path + m)
        data_path.append(data[i, 0])
        if load:
            img = image.load_img(data[i, 0], target_size=(image_width, image_height, 3))
            data_x[i] = image.img_to_array(img)
        data_y[i] = np.array(data[i, 1:1+class_num], dtype="float32")
    data_y = data_y[:, list(np.hstack((low_level, mid_level, high_level)))]
    data_path = np.array(data_path)
    """
    for i in range(class_num):
        alpha[i] += list(data_y[:, i]).count(1.0)
    alpha /= length
    print("The positive ratio of each attribute is:\n", alpha)
    """
    #X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)
    train_cnt = 9500#5700#9500
    val_cnt = 11400#7100#11400
    if load:
        X_train = data_x[:train_cnt]
        X_test = data_x[train_cnt:val_cnt]
    X_train_path = data_path[:train_cnt]
    X_test_path = data_path[train_cnt:val_cnt]
    y_train = data_y[:train_cnt]#, len(low_level)+len(mid_level):
    y_test = data_y[train_cnt:val_cnt]#, len(low_level)+len(mid_level):
    alpha = np.sum(y_train, axis=0)#(len(data_y[0]), )
    alpha /= len(y_train)
    if args.model == "hiarGoogLeNet_high":
        y_train = y_train[:, len(low_level)+len(mid_level):] 
        y_test = y_test[:, len(low_level)+len(mid_level):]
    elif args.model == "hiarGoogLeNet_mid":
        y_train = y_train[:, len(low_level):len(low_level)+len(mid_level)] 
        y_test = y_test[:, len(low_level):len(low_level)+len(mid_level)]
    elif args.model == "hiarGoogLeNet_low":
        y_train = y_train[:, :len(low_level)] 
        y_test = y_test[:, :len(low_level)]
    if load:
        print("The shape of the X_train is: ", X_train.shape)
    print("The shape of the y_train is: ", y_train.shape)
    if load:
        print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    print(alpha)
    #np.save("../results/" + args.model + '_' + save_name + "_X_test.npy", X_test)
    #np.save("../results/" + args.model + '_' + save_name + "_y_test.npy", y_test)
    
    
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
        loss_func = weighted_binary_crossentropy(alpha)
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
        model = hiarBayesGoogLeNet.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_func = weighted_binary_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "hiarBayesInception_v4":
        model = hiarBayesInception_v4(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_func = weighted_binary_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
        metrics = [weighted_acc]
    elif args.model == "hiarBayesResNet":
        model = hiarBayesResNet.build([len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_func = weighted_binary_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
        metrics = [weighted_acc]
        metrics = [mA, 'accuracy']
    gpus_num = len(args.gpus.split(','))
    if gpus_num > 1:
        multi_gpu_model(model, gpus=gpus_num)
    #model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    #opt_sgd = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #opt_sgd = RMSprop(lr=0.001, rho=0.9)
    opt_sgd = Adagrad(lr=0.01)###***
    opt_sgd = Adadelta(lr=1.0, rho=0.95)###***
    #opt_sgd = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999)###***
    #opt_sgd = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)###***
    model.compile(loss=loss_func, optimizer=opt_sgd, loss_weights=loss_weights, metrics=metrics)
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
    if args.model == "hiarGoogLeNetSPP":
        model_dir = 'hiarGoogLeNetSPP_PETA'
    elif args.model == "hiarGoogLeNetWAM":
        model_dir = 'hiarGoogLeNetWAM_PETA'
    elif args.model == "hiarGoogLeNet":
        model_dir = 'hiarGoogLeNet_PETA'
    elif args.model == "hiarBayesGoogLeNet":
        model_dir = 'hiarBayesGoogLeNet_PETA'
        #save_name = 'rl_binary61_multi'#_loss2_ifelse
    elif args.model == "hiarBayesInception_v4":
        model_dir = "hiarBayesInceptionV4_PETA"
    elif args.model == "hiarGoogLeNet_high":
        model_dir = 'hiarGoogLeNet_PETA'
        save_name = 'binary61_high'
    elif args.model == "hiarGoogLeNet_mid":
        model_dir = 'hiarGoogLeNet_PETA'
        save_name = 'binary61_mid_duan'
    elif args.model == "hiarGoogLeNet_low":
        model_dir = 'hiarGoogLeNet_PETA'
        save_name = 'binary61_low_duan'
    elif args.model == "hiarBayesResNet":
        model_dir = 'hiarBayesResNet_PETA'
    checkpointer = ModelCheckpoint(filepath = '../models/imagenet_models/' + model_dir + '/' + save_name+ '_epoch{epoch:02d}_valloss{'+ monitor + ':.2f}.hdf5',
                                   monitor = monitor,
                                   verbose=1, 
                                   save_best_only=True, 
                                   save_weights_only=True,
                                   mode='auto', 
                                   period=1)
    csvlog = CSVLogger('../models/imagenet_models/' + model_dir + '/' + save_name+'_'+str(args.iteration)+'iter'+'_log.csv')#, append=True
    if args.weight != '':
        model.load_weights(args.weight, by_name=True)
    #print(train_generator)
    """
            initial_epoch = 100,
    """
    model.fit_generator(train_generator,
            steps_per_epoch = int(X_train_path.shape[0] / (batch_size * gpus_num)),
            epochs = nb_epoch,
            validation_data = val_generator,
            validation_steps = int(X_test_path.shape[0] / (batch_size * gpus_num)),
            callbacks = [checkpointer, csvlog])
    model.save_weights('../models/imagenet_models/' + model_dir + '/' + save_name+ '_final'+str(args.iteration)+'iter_model.h5')
        #model_pred.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        #model_pred.load_weights('../models/imagenet_models/' + model_dir + '/train_final_model.h5', by_name=True)
        #model_pred.save('../models/imagenet_models/' + model_dir + '/final_model.h5')
