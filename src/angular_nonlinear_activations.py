import random
import tensorflow as tf
import os
import numpy as np
import math
from keras import backend as K

def linear_angular_activation(theta, is_cos=True):
    
    return theta

def cosine_angular_activation(theta, is_cos=True):
    if not is_cos:
        theta = K.cos(theta)
    return theta

def sigmoid_angular_activation(theta, k=1, is_cos=True):
    
    return theta

def square_cosine_angular_activation(theta, is_cos=True):
    if not is_cos:
        theta = K.cos(theta)
    theta = K.sign(theta) * K.pow(theta, 2)
    return theta