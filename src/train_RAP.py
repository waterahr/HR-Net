"""
python train_RAP.py -m GoogLeNet -b 64 -g 1 -w ../models/imagenet_models/GoogLeNet_RAP
python train_RAP.py -m GoogLeNet -c 51 -b 64 -g 1 -w ../models/imagenet_models/GoogLeNet_RAP
python train_RAP.py -m GoogLeNet -c 51 -b 32 -g 0 -w ../models/imagenet_models/GoogLeNet_RAP
python train_RAP.py -m GoogLeNet -c 51 -b 32 -g 0 -hg 160 -wd 75
python train_RAP.py -m GoogLeNet -c 51 -b 64 -g 1 -p 
python train_RAP.py -m Inception_v4 -c 51 -b 32 -wd 299 -hg 299 -i 200 -g 
python train_RAP.py -m GoogLeNetGAP -c 51 -b 32 -i 200 -g 
"""
from network.GoogleLenet import GoogLeNet
from network.GoogLeNetv2 import GoogLeNet as GoogLeNetv2
from network.GoogLenetGAP import GoogLeNetGAP
from network.Inception_v4 import Inception_v4
from keras.optimizers import Adam
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
from angular_losses import weighted_binary_crossentropy

alpha = []

def multi_generator(generator):
    while True:
        x, y = generator.next()
        y_list = []
        for i in range(y.shape[1]):
            y_list.append(y[:, i])
        yield x, y_list

def weighted_acc(y_true, y_pred):
    return K.mean(K.mean(K.equal(y_true, K.round(y_pred)), axis=-1), axis=-1)

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
    """
    #print(K.int_shape(y_true))
    P = K.sum(y_true, axis=-1) + K.epsilon()
    #print("P", P)
    N = K.sum(1-y_true, axis=-1) + K.epsilon()
    #print("N", N)
    TP = K.sum(y_pred * y_true, axis=-1)
    #print("TP", TP)
    TN = K.sum(K.cast_to_floatx(y_pred + y_true == 0))
    #print("TN", TN)
    return K.mean(TP / P + TN / N) / 2

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
    models = ['GoogLeNet', 'Inception_v4', 'GoogLeNetGAP']
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
    parser.add_argument('-i', '--iteration', type=int, default=50,
                        help='The model iterations')
    parser.add_argument('-s', '--split', type=int, default=0,
                        help='The model partion')
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args


    


if __name__ == "__main__":
    #"""
    args = parse_arg()
    save_name = str(args.height) + "x" + str(args.width) + "binary51_newlossnoexp"
    #save_name = "binary3_b2(32)_lr0.0002"
    #part = [2,11,24]
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
    #filename = r"../results/myRAP_labels_pd.csv"
    data = np.array(pd.read_csv(filename))[:, 1:]
    length = len(data)
    #global alpha
    load = False
    if load:
        data_x = np.zeros((length, image_height, image_width, 3))
    data_y = np.zeros((length, class_num))
    data_path = []
    for i in range(length):
        #img = image.load_img(path + m)
        data_path.append(data[i, 0])
        if load:
            img = image.load_img(data[i, 0], target_size=(image_height, image_width, 3))
            data_x[i] = image.img_to_array(img)
        data_y[i] = np.array(data[i, 1:1+class_num], dtype="float32")
    data_path = np.array(data_path)
    #X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)
    split = np.load('../results/RAP_partion.npy').item()
    if load:
        X_train = data_x[list(split['train'][args.split])]#[:26614]
    X_train_path = data_path[list(split['train'][args.split])]
    if load:
        X_test = data_x[list(split['test'][args.split])]
    X_test_path = data_path[list(split['test'][args.split])]
    y_train = data_y[list(split['train'][args.split])]#[:26614]#, len(low_level)+len(mid_level):
    y_test = data_y[list(split['test'][args.split])]#, len(low_level)+len(mid_level):
    alpha = np.sum(y_train, axis=0)#(len(data_y[0]), )
    alpha /= len(y_train)
    """
    y_train_list = []
    y_test_list = []
    for i in range(class_num):
        y_train_list.append(y_train[:, i])
        y_test_list.append(y_test[:, i])
    X_train = data_x[:26614]
    X_test = data_x[26614:33268]
    y_train = data_y[:26614]#, len(low_level)+len(mid_level):
    y_test = data_y[26614:33268]#, len(low_level)+len(mid_level):
    """
    if load:
        print("The shape of the X_train is: ", X_train.shape)
    print("The shape of the y_train is: ", y_train.shape)
    if load:
        print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    print(alpha)
    #alpha = np.ones((1, class_num))
    
    
    #googleNet默认输入32*32的图片
    if args.model == "GoogLeNet":
        model = GoogLeNet.build(image_height, image_width, 3, class_num)
        loss_func = weighted_binary_crossentropy(alpha)
        #loss_func = 'binary_crossentropy'
        #loss_func = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        loss_weights = None
        #metrics=['accuracy']
        metrics = [weighted_acc]
    elif args.model == "GoogLeNetGAP":
        model = GoogLeNetGAP.build(image_height, image_width, 3, class_num)
        loss_func = weighted_binary_crossentropy(alpha)
        #loss_func = 'binary_crossentropy'
        #loss_func = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        loss_weights = None
        #metrics=['accuracy']
        metrics = [weighted_acc]
        metrics = [mA]
    elif args.model == "Inception_v4":
        model = Inception_v4(image_height, image_width, 3, class_num)
        loss_func = weighted_binary_crossentropy(alpha)
        #loss_func = 'binary_crossentropy'
        #loss_func = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        loss_weights = None
        #metrics=['accuracy']
        #metrics = [weighted_acc]
        metrics = [mA]
    elif args.model == "GoogLeNetv2":
        model = GoogLeNetv2.build(image_height, image_width, 3, class_num)
        #loss_func = weighted_binary_crossentropy(alpha)
        loss_func = 'binary_crossentropy'
        #loss_func = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        loss_weights = None
        """
        loss_func = {}
        loss_weights = {}
        for i in range(1, class_num+1):
            loss_func['dense_2_'+str(i)] = 'binary_crossentropy'
            loss_weights['dense_2_'+str(i)] = 1.
        """
        metrics=['accuracy']
        #metrics = [weighted_acc]
    gpus_num = len(args.gpus.split(','))
    if gpus_num > 1:
        multi_gpu_model(model, gpus=gpus_num)
    #model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model.compile(loss=loss_func, optimizer='adam', loss_weights=loss_weights, metrics=metrics)
    #adam = Adam(lr=0.0002)
    #model.compile(loss=loss_func, optimizer=adam, loss_weights=loss_weights, metrics=metrics)
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
    #train_generator = datagen.flow(X_train, y_train_list, batch_size=batch_size)
    #val_generator = datagen.flow(X_test, y_test_list, batch_size=batch_size)
    monitor = 'val_mA'
    if args.model == "GoogLeNet":
        model_dir = 'GoogLeNet_RAP'
    elif args.model == "Inception_v4":
        model_dir = "InceptionV4_RAP"
    elif args.model == "GoogLeNetGAP":
        model_dir = "GoogLeNetGAP_RAP"
    elif args.model == "GoogLeNetv2":
        model_dir = 'GoogLeNet_RAP'
        save_name = "binary51_b4_75v2_"
    checkpointer = ModelCheckpoint(filepath = '../models/imagenet_models/' + model_dir + '/' + save_name+ '_epoch{epoch:02d}_valmA{'+ monitor + ':.2f}.hdf5',
                                   monitor = monitor,
                                   verbose=1, 
                                   save_best_only=True, 
                                   save_weights_only=True,
                                   mode='max', 
                                   period=1)
    csvlog = CSVLogger('../models/imagenet_models/' + model_dir + '/' + save_name+'_'+str(args.iteration)+'iter'+'_log.csv')#, append=True
    if args.weight != '':
        model.load_weights(args.weight, by_name=True)
    print(train_generator)
    #multi_generator(train_generator),multi_generator(val_generator),
    model.fit_generator(train_generator,
            steps_per_epoch = int(X_train_path.shape[0] / (batch_size * gpus_num)),
            epochs = nb_epoch,
            validation_data = val_generator,
            validation_steps = int(X_test_path.shape[0] / (batch_size * gpus_num)),
            callbacks = [checkpointer, csvlog], workers=32)
    model.save_weights('../models/imagenet_models/' + model_dir + '/' + save_name+ '_final_'+str(args.split)+'partion_'+str(args.iteration)+'iter_model.h5')
