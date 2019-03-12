import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from keras.preprocessing import image
from keras.models import Model
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
import cv2
from cv2 import *
import matplotlib.pyplot as plt
import scipy as sp
from scipy.misc import toimage
import sys
sys.path.append("..")
from src.network.GoogleLenet_gap import GoogLeNet
from PIL import Image
import pandas as pd
import tqdm

def cam_from_path(model_gap, img_path, img_height, img_width, weights, is_shown=False):
    img = image.load_img(img_path, target_size=(image_height, image_width, 3))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    #print(img_arr.shape)
    feature_map = model_gap.predict(img_arr)
    #print(feature_map.shape)
    array_feature_map = np.zeros(dtype=np.float32, shape=feature_map.shape[1:3])
    for i in range(0, feature_map.shape[3]):
        array_feature_map += weights[i] * feature_map[0, :, :, i]   
    array_feature_map -= np.average(array_feature_map)
    img_feature_map = Image.fromarray(array_feature_map.astype('uint8'))
    img_feature_map = img_feature_map.resize((image_width, image_height), Image.ANTIALIAS)
    if is_shown:
        plt.imshow(img_feature_map)
    return array_feature_map, np.asarray(img_feature_map)

def cam_from_array(model_gap, img_arr, img_height, img_width, weights, is_shown=False):
    img_arr = np.expand_dims(img_arr, axis=0)
    #print(img_arr.shape)
    feature_map = model_gap.predict(img_arr)
    #print(feature_map.shape)
    array_feature_map = np.zeros(dtype=np.float32, shape=feature_map.shape[1:3])
    for i in range(0, feature_map.shape[3]):
        array_feature_map += weights[i] * feature_map[0, :, :, i]   
    array_feature_map -= np.average(array_feature_map)
    img_feature_map = Image.fromarray(array_feature_map.astype('uint8'))
    img_feature_map = img_feature_map.resize((image_width, image_height), Image.ANTIALIAS)
    if is_shown:
        plt.imshow(img_feature_map)
    return array_feature_map, np.asarray(img_feature_map)

attributes_list = ['upperBodyLogo', 'lowerBodyThinStripes', 'upperBodyThinStripes', 'upperBodyThickStripes', 'accessoryHeadphone', 'carryingBabyBuggy', 'carryingBackpack', 'hairBald', 'footwearBoots', 'carryingOther', 'carryingShoppingTro', 'carryingUmbrella', 'carryingFolder', 'accessoryHairBand', 'accessoryHat', 'lowerBodyHotPants', 'upperBodyJacket', 'lowerBodyJeans', 'accessoryKerchief', 'footwearLeatherShoes', 'hairLong', 'lowerBodyLongSkirt', 'upperBodyLongSleeve', 'carryingLuggageCase', 'carryingMessengerBag', 'accessoryMuffler', 'accessoryNothing', 'carryingNothing', 'upperBodyNoSleeve', 'carryingPlasticBags', 'footwearSandals', 'footwearShoes', 'hairShort', 'lowerBodyShorts', 'upperBodyShortSleeve', 'lowerBodyShortSkirt', 'footwearSneakers', 'footwearStocking', 'upperBodySuit', 'carryingSuitcase', 'lowerBodySuits', 'accessorySunglasses', 'upperBodySweater', 'lowerBodyTrousers', 'upperBodyTshirt', 'upperBodyOther', 'upperBodyVNeck', 'personalLess15', 'personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'lowerBodyCapri', 'lowerBodyCasual', 'upperBodyCasual', 'personalFemale', 'lowerBodyFormal', 'upperBodyFormal', 'lowerBodyPlaid', 'personalMale', 'upperBodyPlaid']
print(attributes_list)

image_width = 75
image_height = 160
class_num = len(attributes_list)
model = GoogLeNet.build(image_height, image_width, 3, class_num)
model.load_weights("../models/imagenet_models/GoogLeNetGAP_PETA/binary61_final50iter_model.h5")
model.summary()

model_gap = Model(inputs=model.input, outputs=model.get_layer('concatenate_9').output)

#print(model.get_layer('dense_1').get_weights())
"""
dense1_weights = np.asarray(model.get_layer('dense_1').get_weights()[0])
print(dense1_weights.shape)
dense2_weights = np.asarray(model.get_layer('dense_2').get_weights()[0])
print(dense2_weights.shape)
"""
weights = np.asarray(model.get_layer('dense_2').get_weights()[0])#np.dot(dense1_weights, dense2_weights)
print(weights.shape)

filename = r"../results/PETA.csv"
data = np.array(pd.read_csv(filename))[:, 1:]
length = len(data)
data_x = np.zeros((length, image_height, image_width, 3))
data_y = np.zeros((length, class_num))
for i in tqdm.tqdm(range(length)):
    #img = image.load_img(path + m)
    img = image.load_img(data[i, 0], target_size=(image_height, image_width, 3))
    data_x[i] = image.img_to_array(img)
    data_y[i] = np.array(data[i, 1:1+61], dtype="float32")
print("The shape of the X is: ", data_x.shape)
print("The shape of the y is: ", data_y.shape)
predictions_prob = model.predict(data_x)
print(predictions_prob.shape)

## 5*3 for all dataset weighted by prediction prob
data_cam = np.zeros((class_num, 5, 3))
data_cam_train = np.zeros((class_num, 5, 3))
for i in tqdm.tqdm(range(length)):
    for attribute in range(data_y.shape[1]):
        if i==11400: data_cam_train[attribute] = data_cam[attribute]
        data_cam[attribute] += predictions_prob[i][attribute] * cam_from_array(model_gap, data_x[i], image_height, image_width, weights[:, attribute])[0]
np.save("../results/orig_feature_maps_dataset_weighted_by_predprob.npy", data_cam)
np.save("../results/orig_feature_maps_trainset_weighted_by_predprob.npy", data_cam_train)
print("orig_feature_maps_weighted_by_predprob")

## 5*3 for all dataset weighted by label
data_cam = np.zeros((class_num, 5, 3))
data_cam_train = np.zeros((class_num, 5, 3))
for i in tqdm.tqdm(range(length)):
    for attribute in range(data_y.shape[1]):
        if i==11400: data_cam_train[attribute] = data_cam[attribute]
        data_cam[attribute] += data_y[i][attribute] * cam_from_array(model_gap, data_x[i], image_height, image_width, weights[:, attribute])[0]
np.save("../results/orig_feature_maps_dataset_weighted_by_label.npy", data_cam)
np.save("../results/orig_feature_maps_trainset_weighted_by_label.npy", data_cam_train)
print("orig_feature_maps_weighted_by_label")

## 160*75 for all dataset weighted by prediction prob
data_cam = np.zeros((class_num, 160, 75))
data_cam_train = np.zeros((class_num, 160, 75))
for i in tqdm.tqdm(range(length)):
    for attribute in range(data_y.shape[1]):
        if i == 11400: data_cam_train[attribute] = data_cam[attribute]
        data_cam[attribute] += predictions_prob[i][attribute] * cam_from_array(model_gap, data_x[i], image_height, image_width, weights[:, attribute])[1]
np.save("../results/resize_feature_maps_dataset_weighted_by_predprob.npy", data_cam)
np.save("../results/resize_feature_maps_trainset_weighted_by_predprob.npy", data_cam_train)
print("resize_feature_maps_weighted_by_predprob")

## 160*75 for all dataset weighted by label
data_cam = np.zeros((class_num, 160, 75))
data_cam_train = np.zeros((class_num, 160, 75))
for i in tqdm.tqdm(range(length)):
    for attribute in range(data_y.shape[1]):
        if i == 11400: data_cam_train[attribute] = data_cam[attribute]
        data_cam[attribute] += data_y[i][attribute] * cam_from_array(model_gap, data_x[i], image_height, image_width, weights[:, attribute])[1]
np.save("../results/resize_feature_maps_dataset_weighted_by_label.npy", data_cam)
np.save("../results/resize_feature_maps_trainset_weighted_by_label.npy", data_cam_train)
print("resize_feature_maps_weighted_by_label")