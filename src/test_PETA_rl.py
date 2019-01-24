"""
python test_PETA_rl.py -m mlp_labels -g 0 -r ma_as_reward_one_choose -w
"""
import os
import argparse
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from network.hiarBayesGoogLenet import hiarBayesGoogLeNet

def parse_arg():
    inputs = ['mlp_labels', 'lstm_labels']
    rewards_type = ['ma_as_reward_one_choose', 'pn_as_reward_one_choose', 'ma_as_reward_while_choose', 'pn_as_reward_while_choose', 'cospn_as_reward_one_choose', 'cospn_as_reward_while_choose']
    parser = argparse.ArgumentParser(description='training of the RL version of the HR-Net...')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='The gpu device\'s ID need to be used')
    parser.add_argument('-a', '--actions', type=int, default=2,
                        help='The n_actions that the RL_brain')
    parser.add_argument('-f', '--features', type=int, default=4,
                        help='The n_features that the RL_brain')
    parser.add_argument('-m', '--models', type=str, default='mlp_labels',
                        help='The inputs of the RL_brain, including: '+str(inputs))
    parser.add_argument('-w', '--weights', type=int, default=1,
                        help='The next learning step of the RL_brain')
    parser.add_argument('-r', '--rewards', type=str, default='',
                        help='The type of the RL_brain rewards, including: '+str(rewards_type))
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

def mA(y_pred, y_true):
    M = len(y_pred)
    L = len(y_pred[0])
    res = 0
    for i in range(L):
        P = sum(y_true[:, i])
        N = M - P
        TP = sum(y_pred[:, i]*y_true[:, i])
        TN = list(y_pred[:, i]+y_true[:, i] == 0).count(True)
        #print(P,',', N,',', TP,',', TN)
        if P != 0:
            res += TP/P + TN/N
        else:
            res += TN/N
    return res / (2*L)

def train(observation, action, save_name, batch_size_p=32, nb_epoch_p=50, ma_as_reward=True):
    #print(int(save_name[save_name.rindex('_')+1:]))
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
    image_width = 160
    image_height = 75
    class_num = action.shape[0]
    filename = r"../results/PETA.csv"
    data = np.array(pd.read_csv(filename))[:, 1:]
    length = len(data)
    data_x = np.zeros((length, image_width, image_height, 3))
    data_y = np.zeros((length, class_num))
    low_level = []
    mid_level = []
    high_level = []
    for i in range(action.shape[0]):
        if action[i] == 0:
            low_level.append(i)
        elif action[i] == 1:
            mid_level.append(i)
        elif action[i] == 2:
            high_level.append(i)
        else:
            print("ERROR ACTION!!!")
    print(low_level)
    print(mid_level)
    print(high_level)
    labels_list_file = r"/home/anhaoran/data/pedestrian_attributes_PETA/PETA/labels.txt" 
    labels_list_data = open(labels_list_file)
    lines = labels_list_data.readlines()
    tmp_list = []
    for line in lines:
        dat = line.split()
        #print(attr)
        tmp_list.append(dat[1])
    #attributes_list = list(np.array(tmp_list))
    attributes_list = list(np.array(tmp_list)[[0,1,2,3,4,5,6,7,8,9,27,32,40,50,56]])
    print("-------------------------------------------------------------------------")
    print("***Low level attributes: ")
    for i in low_level:
        print(attributes_list[i], end=", ")
    print("\n***Medium level attributes: ")
    for i in mid_level:
        print(attributes_list[i], end=", ")
    print("\n***High level attributes: ")
    for i in high_level:
        print(attributes_list[i], end=", ")
    print()
    print("-------------------------------------------------------------------------")
    if np.shape(low_level)[0] == 0 or np.shape(mid_level)[0] == 0 or np.shape(high_level)[0] == 0:
        reward = -1
        return action, reward, reward>=0.9 or int(save_name[save_name.rindex('_')+1:]) >= 10
    for i in range(length):
        #img = image.load_img(path + m)
        img = image.load_img(data[i, 0], target_size=(image_width, image_height, 3))
        data_x[i] = image.img_to_array(img)
        #data_y[i] = np.array(data[i, 1:1+class_num], dtype="float32")
        data_y[i] = np.array(data[i, [1,2,3,4,5,6,7,8,9,10,28,33,41,51,57]], dtype="float32")
    data_y = data_y[:, list(np.hstack((low_level, mid_level, high_level)))]
    X_train = data_x[:9500]
    X_test = data_x[9500:11400]
    y_train = data_y[:9500]#, len(low_level)+len(mid_level):
    y_test = data_y[9500:11400]#, len(low_level)+len(mid_level):
    XX = data_x[11400:]
    yy = data_y[11400:]
    print("The shape of the X_train is: ", X_train.shape)
    print("The shape of the y_train is: ", y_train.shape)
    print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    
    model = hiarBayesGoogLeNet.build(image_width, image_height, 3, [len(low_level), len(mid_level), len(high_level)])
    model.compile(loss='binary_crossentropy', optimizer='adam', loss_weights=None, metrics=['accuracy'])
    
    batch_size = batch_size_p
    nb_epoch = nb_epoch_p
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = datagen.flow(X_test, y_test, batch_size=batch_size)
    model.fit_generator(train_generator,
            steps_per_epoch = int(X_train.shape[0] / batch_size),
            epochs = nb_epoch,
            validation_data = val_generator,
            validation_steps = int(X_test.shape[0] / batch_size))
    #model.save_weights('../models/imagenet_models/' + save_name + '_final_model.h5')
    
    predictions_prob = model.predict(XX)
    predictions = np.array(predictions_prob >= 0.5, dtype="float64")
    label = yy
    reward = mA(predictions, label)
    return action, reward, True

args = parse_arg()
n_actions = args.actions
n_features = args.features

if args.models == 'mlp_labels':
    from rl.RL_brain_mlp_labels import PolicyGradient
    n_actions = 3
    #n_features = 61
    n_features = 15
    #weight_file = "../models/imagenet_models/rl/ma_as_reward_one_choose/RL_brain_mlp_labels-step"+str(args.weights)+"-1"
    weight_file = "../models/imagenet_models/rl_test15/"+args.rewards+"/RL_brain_mlp_labels-step"+str(args.weights)+"-1"
    RL = PolicyGradient(
            n_actions=n_actions,
            n_features = n_features,
            learning_rate = 0.02,
            reward_decay = 0.99,
            weights = weight_file,
            init_step = args.weights,
            reward_type = args.rewards
        )

    observation = np.ones(n_features)
    if args.rewards == "ma_as_reward_one_choose" or args.rewards == "pn_as_reward_one_choose" or args.rewards == "cospn_as_reward_one_choose":
        action = RL.choose_action(observation)
    elif args.rewards == "ma_as_reward_while_choose" or args.rewards == "pn_as_reward_while_choose"or args.rewards == "cospn_as_reward_while_choose":
        #"""
        obser = observation
        while True:
            print(".", end="")
            action = RL.choose_action(obser)#print(np.shape(action))###(61,)
            if sum(np.array(action)-obser) == 0:
                print()
                break
            obser = np.array(action)
    _, reward, _ = train(observation, np.array(action), "rl/mlp_test")
    print("Reward(mA): ", reward)
                
elif args.models == 'lstm_labels':
    from rl.RL_brain_lstm_labels import PolicyGradient
    n_actions = 3
    #n_features = 61
    n_features = 15
    RL = PolicyGradient(
            n_actions=n_actions,
            n_features = n_features,
            learning_rate = 0.02,
            reward_decay = 0.99,
            weights = "../models/imagenet_models/rl_lstm/RL_brain_lstm_labels-step"+str(args.weights-1)+"-1",
            init_step = args.weights
        )

    observation = np.ones((n_features, n_actions))
    while True:
        print(".", end="")
        action = RL.choose_action(observation)#print(np.shape(action))###(61,)
        tmp = (np.arange(n_actions) == np.array(action)[:, None]).astype(np.int32)
        if sum(sum(tmp-observation)) == 0:
            print()
            break
        observation = tmp
    _, reward, _ = train(observation, np.array(action), "rl_lstm/lstm_test")
    print("Reward(mA): ", reward)