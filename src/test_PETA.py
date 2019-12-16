"""
python test_PETA.py -g 0 -c 61 -b 256  -m GoogLeNetSPP -w ../models/
python test_PETA.py -g 0 -c 61 -b 256  -m GoogLeNet -w ../models/
python test_PETA.py -g 0 -c 61 -b 256  -m GoogLeNet -w ../models/imagenet_models/GoogLeNet_PETA/binary61_depth
python test_PETA.py -g 1 -c 68 -b 256 -m OEDCGoogLeNetSPP -w ../models/xxxxx.hdf5
python test_PETA.py -g 1 -c 14 -b 64 -m OEDCGoogLeNetSPP_lowerBody  -w ../models/xxxxx.hdf5
"""
from network.GoogLenetSPP import GoogLeNetSPP
from network.GoogleLenet import GoogLeNet
from network.Inception_v4 import Inception_v4
from network.GoogleLenet_gap import GoogLeNet as GoogLeNet_gap
from network.OEDC_GoogLenetSPP import OEDCGoogLeNetSPP
from network.OEDC_GoogLenetSPP_lowerBody import OEDCGoogLeNetSPP_lowerBody
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from keras import Model, Sequential
import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import re
import tqdm
from keras import backend as K
from angular_losses import weighted_categorical_crossentropy, coarse_to_fine_categorical_crossentropy_lowerbody

alpha = []

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
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args


    


if __name__ == "__main__":
    save_name = "binary61"
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
            samplewise_std_normalization=False)
    else:
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False)
    image_width = args.width
    image_height = args.height
    filename = r"../results/PETA.csv"
    data = np.array(pd.read_csv(filename))[:, 1:]
    length = len(data)
    data_x = np.zeros((length, image_width, image_height, 3))
    data_y = np.zeros((length, class_num))
    for i in range(11400, length):
        #img = image.load_img(path + m)
        img = image.load_img(data[i, 0], target_size=(image_width, image_height, 3))
        data_x[i] = image.img_to_array(img)
        data_y[i] = np.array(data[i, 1:1+class_num], dtype="float32")
    X_test = data_x[11400:]
    y_test = data_y[11400:]
    print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    
    
    #googleNet默认输入32*32的图片
    if args.model == "GoogLeNetSPP":
        model_dir = "GoogLeNetSPP_PETA/"
        model = GoogLeNetSPP.build(None, None, 3, class_num)
        loss_func = 'binary_crossentropy'
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "GoogLeNet":
        model_dir = "GoogLeNet_PETA/"
        model = GoogLeNet.build(image_width, image_height, 3, class_num, model_depth=args.depth)
        #model = Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
        loss_func = 'binary_crossentropy'
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "Inception_v4":
        model_dir = "InceptionV4_PETA/"
        model = Inception_v4(image_width, image_height, 3, class_num)
        loss_func = 'binary_crossentropy'
        loss_weights = None
        metrics=['accuracy']
    elif args.model == "GoogLeNet_gap":
        model = GoogLeNet_gap.build(image_width, image_height, 3, class_num, model_depth=args.depth)
        #model = Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
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
    
    reg = args.weight + "_(e|f)1*"
    print(reg)
    weights = [s for s in os.listdir("../models/imagenet_models/" + model_dir) 
          if re.match(reg, s)]
    print(weights)
    for w in tqdm.tqdm(weights):
        model.load_weights("../models/imagenet_models/" + model_dir + w, by_name=True)
        predictions = model.predict(X_test)
        print("The shape of the predictions_test is: ", predictions.shape)
        np.save("../results/predictions/" + args.model + "_" + w + ".npy", predictions)
        print("../results/predictions/" + args.model + "_" + w + ".npy")
    #np.save("../results/predictions/" + args.model+'_depth'+str(args.depth) + '_' + save_name + "_predictions50_imagenet_test7600.npy", predictions)