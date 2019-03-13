"""
python train_RAP_hiarchical.py -m hiarBayesGoogLeNet -b 64 -g 1 -w ../models/imagenet_models/hiarBayesGoogLeNet_RAP
python train_RAP_hiarchical.py -m hiarGoogLeNet -b 64 -g 1 -w ../models/imagenet_models/hiarGoogLeNet_RAP
python train_RAP_hiarchical.py -m hiarBayesGoogLeNet -b 64 -c 51 -wd 75 -hg 160 -i 500 -g 1 -w ../models/imagenet_models/hiarBayesGoogLeNet_RAP
python train_RAP_hiarchical.py -m hiarGoogLeNet -b 64 -g 1 -c 51 -w ../models/imagenet_models/hiarGoogLeNet_RAP

python train_RAP_hiarchical.py -g 1 -m hiarGoogLeNet -wd 227 -hg 227 -b 32 -c 51 -i 200
python train_RAP_hiarchical.py -g 1 -m hiarBayesGoogLeNet -wd 227 -hg 227 -b 64 -c 51 -i 200
python train_RAP_hiarchical.py -g 1 -m hiarBayesGoogLeNet -wd 227 -hg 227 -b 64 -c 92 -i 200
python train_RAP_hiarchical.py -g 1 -m hiarGoogLeNet -wd 227 -hg 227 -b 64 -c 92 -i 200
python train_RAP_hiarchical.py -g 1 -m hiarBayesGoogLeNet_inception -wd 227 -hg 227 -b 64 -c 92 -i 200
python train_RAP_hiarchical.py -g 1 -m hiarGoogLeNet -wd 227 -hg 227 -b 128 -c 51 -i 1000 -w 
python train_RAP_hiarchical.py -g 1 -m hiarGoogLeNet_inception -wd 227 -hg 227 -b 128 -c 51 -i 1000
ls
python train_RAP_hiarchical.py -m hiarBayesInception_v4 -wd 299 -hg 299 -b 16 -c 51 -i 200 -g 
python train_RAP_hiarchical.py -m hiarGoogLeNetGAP -b 128 -c 51 -i 200 -g 
python train_RAP_hiarchical.py -m hiarBayesGoogLeNetGAP -b 128 -c 51 -i 200 -g 
"""
#from network.hiarGoogLenetSPP import hiarGoogLeNetSPP
#from network.hiarGoogLenetWAM import hiarGoogLeNetWAM
from network.hiarGoogLenet import hiarGoogLeNet
from network.hiarGoogLenetGAP import hiarGoogLeNetGAP
from network.hiarBayesGoogLenetGAP import hiarBayesGoogLeNetGAP
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
from keras import backend as K
from angular_losses import bayes_binary_crossentropy
from angular_losses import weighted_binary_crossentropy


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

alpha = []

def parse_arg():
    models = ['hiarGoogLeNet', 'hiarGoogLeNetGAP', 'hiarBayesGoogLeNet', 'hiarBayesGoogLeNetGAP', 'hiarBayesGoogLeNet_inception', 'hiarGoogLeNet_inception']
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
    parser.add_argument('-i', '--iteration', type=int, default=100,
                        help='The model iterations')
    parser.add_argument('-s', '--split', type=int, default=0,
                        help='The split')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args


    


if __name__ == "__main__":
    args = parse_arg()
    save_name = str(args.height) + "x" + str(args.width) + "binary51_beforreduction_oldhier_newlossnoexp_split" + str(args.split)
    low_level = [11]#,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91
    mid_level = [9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
    high_level = [0,1,2,3,4,5,6,7,8,43,44,45,46,47,48,49,50]#,51,52,53,54,55,56,57,58,59,60,61,62
    """
    save_name = "227x227binary92_oldhiar_newlossnoexp_split" + str(args.split)
    low_level = [11,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91]#
    mid_level = [9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
    high_level = [0,1,2,3,4,5,6,7,8,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62]#
    """
    """
    save_name = "binary51_beforreduction_newhier_newlossnoexp_split" + str(args.split)
    low_level = [11]#,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91
    mid_level = [4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    high_level = [0,1,2,3]#,51,52,53,54,55,56,57,58,59,60,61,62
    """
    """
    save_name = "binary3"
    low_level = [11]
    mid_level = [24]
    high_level = [2]
    """
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
    load = False
    if load:
        data_x = np.zeros((length, image_height, image_width, 3))
    data_path = []
    data_y = np.zeros((length, class_num))
    for i in range(length):
        #img = image.load_img(path + m)
        data_path.append(data[i, 0])
        if load:
            img = image.load_img(data[i, 0], target_size=(image_height, image_width, 3))
            data_x[i] = image.img_to_array(img)
        data_y[i] = np.array(data[i, 1:1+class_num], dtype="float32")
    data_y = data_y[:, list(np.hstack((low_level, mid_level, high_level)))]
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
    #alpha = np.exp(-alpha)
    print(alpha)
    print(alpha.shape)
    if load:
        print("The shape of the X_train is: ", X_train.shape)
    print("The shape of the y_train is: ", y_train.shape)
    if load: 
        print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    
    
    #googleNet默认输入32*32的图片
    if args.model == "hiarGoogLeNet":
        model = hiarGoogLeNet.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_func = weighted_binary_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
        metrics = [weighted_acc]
    elif args.model == "hiarGoogLeNetGAP":
        model = hiarGoogLeNetGAP.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func = 'binary_crossentropy'#weighted_categorical_crossentropy(alpha)
        loss_func = weighted_binary_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
        metrics = [weighted_acc]
        metrics = [mA, 'accuracy']
    elif args.model == "hiarBayesGoogLeNet":
        model = hiarBayesGoogLeNet.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_func = weighted_binary_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
        metrics = [weighted_acc]
        metrics = [mA, 'accuracy']
    elif args.model == "hiarBayesGoogLeNetGAP":
        model = hiarBayesGoogLeNetGAP.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_func = weighted_binary_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
        metrics = [weighted_acc]
        metrics = [mA, 'accuracy']
    elif args.model == "hiarBayesInception_v4":
        model = hiarBayesInception_v4(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_func = weighted_binary_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
        metrics = [weighted_acc]
        metrics = [mA]
    elif args.model == "hiarBayesGoogLeNet_inception":
        model = hiarBayesGoogLeNet_inception.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_func = weighted_binary_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
        metrics = [weighted_acc]
    elif args.model == "hiarGoogLeNet_inception":
        model = hiarBayesGoogLeNet_inception.build(image_height, image_width, 3, [len(low_level), len(mid_level), len(high_level)])
        loss_func ='binary_crossentropy'#bayes_binary_crossentropy(alpha, y_train)#weighted_categorical_crossentropy(alpha)
        loss_func = weighted_binary_crossentropy(alpha)
        loss_weights = None
        metrics=['accuracy']
        metrics = [weighted_acc]
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
    if load:
        train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    else:
        train_generator = generate_imgdata_from_file(X_train_path, y_train, batch_size, image_height, image_width)
    if load:
        val_generator = datagen.flow(X_test, y_test, batch_size=batch_size)
    else:
        val_generator = generate_imgdata_from_file(X_test_path, y_test, batch_size, image_height, image_width)
    monitor = 'val_mA'
    if args.model == "hiarGoogLeNet":
        model_dir = 'hiarGoogLeNet_RAP'
    elif args.model == "hiarGoogLeNetGAP":
        model_dir = 'hiarGoogLeNetGAP_RAP'
    elif args.model == "hiarBayesGoogLeNet": 
        model_dir = 'hiarBayesGoogLeNet_RAP'
    elif args.model == "hiarBayesGoogLeNetGAP": 
        model_dir = 'hiarBayesGoogLeNetGAP_RAP'
    elif args.model == "hiarBayesInception_v4":
        model_dir = "hiarBayesInceptionV4_RAP"
    elif args.model == "hiarBayesGoogLeNet_inception":
        model_dir = 'hiarBayesGoogLeNet_Inception_RAP'
    elif args.model == "hiarGoogLeNet_inception":
        model_dir = 'hiarGoogLeNet_Inception_RAP'
    elif args.model == "hiarBayesGoogLeNetv2":
        model_dir = 'hiarBayesGoogLeNet_RAP'
    checkpointer = ModelCheckpoint(filepath = '../models/imagenet_models/' + model_dir + '/' + save_name+ '_epoch{epoch:02d}_vallossmA{'+ monitor + ':.2f}.hdf5',
                                   monitor = monitor,
                                   verbose=1, 
                                   save_best_only=True, 
                                   save_weights_only=True,
                                   mode='max', 
                                   period=1)
    csvlog = CSVLogger('../models/imagenet_models/' + model_dir + '/' + save_name+'_'+str(args.iteration)+'iter'+'_log.csv', append=True)#
    if args.weight != '':
        model.load_weights(args.weight, by_name=True)
    print(train_generator)
    """
            , initial_epoch = 50,
    """
    ###multi_generator(train_generator),multi_generator(val_generator)
    model.fit_generator(train_generator,
            steps_per_epoch = int(X_train_path.shape[0] / (batch_size * gpus_num)),
            epochs = nb_epoch,
            validation_data = val_generator,
            validation_steps = int(X_test_path.shape[0] / (batch_size * gpus_num)),
            callbacks = [checkpointer, csvlog], workers = 32)#, initial_epoch = 200
    model.save_weights('../models/imagenet_models/' + model_dir + '/' + save_name+ '_final'+str(args.iteration)+'iter_model.h5')
