"""
python extrac_feature.py -g 0,1 -c 61 -w ../models/xxxxx.hdf5 -m GoogLeNetSPP
python extrac_feature.py -g 0,1 -c 68 -w ../models/xxxxx.hdf5 -m OEDCGoogLeNetSPP
"""
from keras import Model, Sequential
from GoogLenetSPP import GoogLeNetSPP
from OEDC_GoogLenetSPP import OEDCGoogLeNetSPP
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
import pickle
from keras import backend as K

def parse_arg():
    models = ['GooLeNetSPP', 'OEDCGoogLeNetSPP']
    parser = argparse.ArgumentParser(description='training of the WPAL...')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='The gpu device\'s ID need to be used')
    parser.add_argument('-c', '--classes', type=int, default=65,
                        help='The total number of classes to be predicted')
    parser.add_argument('-wd', '--width', type=int, default=160,
                        help='The width of thWPAL_PETAe picture')
    parser.add_argument('-hg', '--height', type=int, default=75,
                        help='The height of the picture')
    parser.add_argument('-w', '--weight', type=str, default='',
                        help='The weights file of the pre-training')
    parser.add_argument('-m', '--model', type=str, default='',
                        help='The model including: '+str(models))
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

def weighted_categorical_crossentropy(y_true, y_pred):
    total_num = 19000
    f = open('../results/PETA_ratio_positive_samples_for_attributes.json',"r")
    for line in f:
        ratio = json.loads(line)
    f.close()
    ratio_array = np.array(list(ratio.values())) * 1.0 / total_num
    #print(data)
    #print(K.int_shape(y_pred))(None, 65)
    loss = K.zeros_like(K.categorical_crossentropy(y_true, y_pred))
    for i in range(K.int_shape(y_pred)[1]):
        loss += 0.5 * (y_true[:, i] * K.log(y_pred[:, i]) / ratio_array[i] + y_pred[:, i] * K.log(y_true[:, i]) / (1 - ratio_array[i]))
    return loss
    


if __name__ == "__main__":
    args = parse_arg()
    class_num = args.classes
    #googleNet默认输入32*32的图片
    if args.model == "GoogLeNetSPP":
        model = GoogLeNetSPP.build(None, None, 3, class_num)
    elif args.model == "OEDCGoogLeNetSPP":
        model = OEDCGoogLeNetSPP.build(None, None, 3, 7, [3, 7, 11, 6, 7, 12, 15])#[4, 7, 11, 7, 7, 13, 16]
    model = Model(inputs=model.input,
                  outputs=model.get_layer('concatenate_10').output)
    gpus_num = len(args.gpus.split(','))
    if gpus_num != 1:
        model = multi_gpu_model(model, gpus=gpus_num)
    model.summary()


    
    image_width = args.width
    image_height = args.height
    if args.model == "GoogLeNetSPP":
        filename = r"../results/PETA_labels_pd.csv"
    elif args.model == "OEDCGoogLeNetSPP":
        filename = r"../results/PETA_coarse_to_fine_labels_pd.csv"
    data = np.array(pd.read_csv(filename))[:, 1:]
    length = len(data)
    data_x = np.zeros((length, image_width, image_height, 3))
    data_y = np.zeros((length, class_num))
    for i in range(length):
        #img = image.load_img(path + m)
        img = image.load_img(data[i, 0], target_size=(image_width, image_height, 3))
        data_x[i] = image.img_to_array(img)
        data_y[i] = np.array(data[i, 1:1+class_num], dtype="float32")
    print("The shape of the X_train is: ", data_x.shape)
    print("The shape of the y_train is: ", data_y.shape)


    if args.model == "GoogLeNetSPP":
        model_dir = 'WPAL_PETA'
    elif args.model == "OEDCGoogLeNetSPP":
        model_dir = 'OEDCWPAL_PETA'
    if args.weight != '':
        model.load_weights(args.weight)

    features_all = model.predict(data_x)
    labels_all = data_y
    data_all = {'features_all':features_all, 
                'labels_all':data_y}
    savename = "../results/" + model_dir + '_features_all.pickle'
    savename = "../results/" + model_dir + '_features_all.pickle'
    fsave = open(savename, 'wb')
    pickle.dump(data_all, fsave)
    fsave.close()