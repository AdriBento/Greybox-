from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os
from math import log2
import random
import matplotlib.pyplot as plt

#import network

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:
        w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class:
        propensity_score = freq_class / total_pixels.
    References: https://arxiv.org/abs/1606.02147
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.
    """

    class_count = 0
    total = 0
    for name, _, _,label in dataloader: #for name, _, label,_ in dataloader:
        #print("#####################")
        label = label.cpu().numpy()
        # Flatten label
        flat_label = label.flatten()
        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        #print(class_count)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def make_x_y_data(txt_data_path, shuffled=False):
    data_id, data_categories, data_label = parse_x_y_data(txt_data_path)
    data_hot = []
    for i in range(0, len(data_categories)):
        one_hot_list = make_hot(data_categories[i])
        data_hot.append(one_hot_list)
    if shuffled:
        zipped_list = list(zip(data_hot, data_label))
        random.shuffle(zipped_list)
        data_hot, data_label = zip(*zipped_list)

    return data_id, data_hot, data_label

def make_x_y_data_cdgai(txt_data_path, shuffled=False):
    data_id, data_categories, data_label = parse_x_y_data_cdgai(txt_data_path)
    data_hot = []
    for i in range(0, len(data_categories)):
        one_hot_list = make_hot_cdg(data_categories[i])
        data_hot.append(one_hot_list)
    if shuffled:
        zipped_list = list(zip(data_hot, data_label))
        random.shuffle(zipped_list)
        data_hot, data_label = zip(*zipped_list)

    return data_id, data_hot, data_label

def make_x_y_data_pascal(txt_data_path, shuffled=False):
    data_id, data_categories, data_label = parse_x_y_data_pascal(txt_data_path)
    data_hot = []
    for i in range(0, len(data_categories)):
        one_hot_list = make_hot_pascal(data_categories[i])
        data_hot.append(one_hot_list)
    if shuffled:
        zipped_list = list(zip(data_hot, data_label))
        random.shuffle(zipped_list)
        data_hot, data_label = zip(*zipped_list)

    return data_id, data_hot, data_label

def make_x_y_data_pascal_clean(txt_data_path, shuffled=False):
    data_id, data_categories, data_label = parse_x_y_data_pascal_clean(txt_data_path)
    data_hot = []
    for i in range(0, len(data_categories)):
        one_hot_list = make_hot_pascal_clean(data_categories[i])
        data_hot.append(one_hot_list)
    if shuffled:
        zipped_list = list(zip(data_hot, data_label))
        random.shuffle(zipped_list)
        data_hot, data_label = zip(*zipped_list)

    return data_id, data_hot, data_label

def make_x_y_data_pascal_big_attrib(txt_data_path, shuffled=False):
    data_id, data_categories, data_label = parse_x_y_data_pascal_big_attrib(txt_data_path)
    data_hot = []
    for i in range(0, len(data_categories)):
        one_hot_list = make_hot_pascal_big_attrib(data_categories[i])
        data_hot.append(one_hot_list)
    if shuffled:
        zipped_list = list(zip(data_hot, data_label))
        random.shuffle(zipped_list)
        data_hot, data_label = zip(*zipped_list)

    return data_id, data_hot, data_label

def make_x_y_data_pascal_sur_mesure(txt_data_path, shuffled=False):
    data_id, data_categories, data_label = parse_x_y_data_sur_mesure(txt_data_path)
    data_hot = []
    for i in range(0, len(data_categories)):
        one_hot_list = make_hot_pascal_sur_mesure(data_categories[i])
        data_hot.append(one_hot_list)
    if shuffled:
        zipped_list = list(zip(data_hot, data_label))
        random.shuffle(zipped_list)
        data_hot, data_label = zip(*zipped_list)

    return data_id, data_hot, data_label

def make_x_y_data_pascal_4_classes(txt_data_path, shuffled=False):
    data_id, data_categories, data_label = parse_x_y_data_4_classes(txt_data_path)
    print("Done")
    print(data_id)
    print(data_categories)
    print(data_label)
    data_hot = []
    for i in range(0, len(data_categories)):
        one_hot_list = make_hot_pascal_4_classes(data_categories[i])
        data_hot.append(one_hot_list)
    if shuffled:
        zipped_list = list(zip(data_hot, data_label))
        random.shuffle(zipped_list)
        data_hot, data_label = zip(*zipped_list)

    return data_id, data_hot, data_label

def make_x_y_data_pascal_4_classes_15_attribs(txt_data_path, shuffled=False):
    data_id, data_categories, data_label = parse_x_y_data_4_classes_15_attribs(txt_data_path)
    print("Done")
    print(data_id)
    print(data_categories)
    print(data_label)
    data_hot = []
    for i in range(0, len(data_categories)):
        one_hot_list = make_hot_pascal_4_classes_15_attribs(data_categories[i])
        data_hot.append(one_hot_list)
    if shuffled:
        zipped_list = list(zip(data_hot, data_label))
        random.shuffle(zipped_list)
        data_hot, data_label = zip(*zipped_list)

    return data_id, data_hot, data_label



def make_x_y_data_pascal_proportion(txt_data_path, shuffled=False):
    data_id, data_categories, data_label, data_attrib = parse_x_y_data_pascal_proportion(txt_data_path)
    data_hot = []
    for i in range(0, len(data_categories)):
        one_hot_list = make_hot_pascal_proportion(data_categories[i], data_attrib[i])
        data_hot.append(one_hot_list)
    if shuffled:
        zipped_list = list(zip(data_hot, data_label))
        random.shuffle(zipped_list)
        data_hot, data_label = zip(*zipped_list)

    return data_id, data_hot, data_label

def make_x_y_data_proportion(txt_data_path, shuffled=False):
    data_id, data_categories, data_label = parse_x_y_data_proportion(txt_data_path)
    data_hot = []
    for i in range(0, len(data_categories)):

        one_hot_list = make_hot(data_categories[i])
        one_hot_list = make_hot(data_categories[i])
        data_hot.append(one_hot_list)
    if shuffled:
        zipped_list = list(zip(data_hot, data_label))
        random.shuffle(zipped_list)
        data_hot, data_label = zip(*zipped_list)

    return data_id, data_hot, data_label


def parse_x_y_data(txt_data_path):
    # go from data.txt to 2 vectors containing images and categories. data.txt must be in form "img path" ; "cat 1" ; "cat2 and a vector containing labels
    data_images = []
    data_categories = []
    data_label = []

    data_txt = open(txt_data_path, 'r')
    list_data_txt = [line.split('\n')[0] for line in data_txt.readlines()]
    parsed_list_data_txt = [elem.split(';') for elem in list_data_txt]
    # shuffled_parsed_list_data_txt = sample(parsed_list_data_txt, len(parsed_list_data_txt))

    for i in range(0, len(parsed_list_data_txt)):
        if int(parsed_list_data_txt[i][-1]) < 100:
            data_images.append(parsed_list_data_txt[i][0])  # x_data -> image path
            attrib_index = parsed_list_data_txt[i][1:-1]
            # attrib_index = [str(int(i)-1) for i in attrib_index] #just to correct a shift in the DICT
            data_categories.append(attrib_index)  # x data -> categories
            data_label.append(int(parsed_list_data_txt[i][-1]))  # y data -> labels

    return data_images, data_categories, data_label

def parse_x_y_data_cdgai(txt_data_path):
    # go from data.txt to 2 vectors containing images and categories. data.txt must be in form "img path" ; "cat 1" ; "cat2 and a vector containing labels
    data_images = []
    data_categories = []
    data_label = []

    data_txt = open(txt_data_path, 'r')
    list_data_txt = [line.split('\n')[0] for line in data_txt.readlines()]

    parsed_list_data_txt = [elem.split(',') for elem in list_data_txt[1:]]

    # shuffled_parsed_list_data_txt = sample(parsed_list_data_txt, len(parsed_list_data_txt))
    count = 0
    for i in range(0, len(parsed_list_data_txt)):
        if int(parsed_list_data_txt[i][-1]) < 100:
            data_images.append(parsed_list_data_txt[i][0])  # x_data -> image path
            attrib_index = parsed_list_data_txt[i][1:-1]
            attrib_index[0] = attrib_index[0][-1]
            attrib_index[-1] = attrib_index[-1][1]
            attrib_index = [int(j) - 1 for j in attrib_index]

            # attrib_index = [str(int(i)-1) for i in attrib_index] #just to correct a shift in the DICT
            data_categories.append(attrib_index)  # x data -> categories
            data_label.append(int(parsed_list_data_txt[i][-1]))  # y data -> labels


    return data_images, data_categories, data_label

def parse_x_y_data_pascal(txt_data_path):
    # go from data.txt to 2 vectors containing images and categories. data.txt must be in form "img path" ; "cat 1" ; "cat2 and a vector containing labels
    data_images = []
    data_classes = []
    data_attrib = []

    data_txt = open(txt_data_path, 'r')
    list_data_txt = [line.split('\n')[0] for line in data_txt.readlines()]

    parsed_list_data_txt = [elem.split(';') for elem in list_data_txt]
    # shuffled_parsed_list_data_txt = sample(parsed_list_data_txt, len(parsed_list_data_txt))

    for i in range(0, len(parsed_list_data_txt)):
        data_images.append(parsed_list_data_txt[i][0])  # x_data -> image path
        data_classes.append(int(parsed_list_data_txt[i][2]))
        print(int(parsed_list_data_txt[i][2]))
        data_attrib.append([i.split(',')[-1] for i in parsed_list_data_txt[i][4:]])

    return data_images, data_attrib, data_classes

def parse_x_y_data_pascal_clean(txt_data_path):
    ELEMENTS_CLEAN = {'Bird': 0, 'Aeroplane': 1, 'Cat': 2, 'Dog': 3, 'Sheep': 4, 'Train': 5, 'Bicycle': 6, 'Horse': 7,
                      'Bottle': 8, 'Person': 9, 'Car': 10, 'Pottedplant': 11, 'Motorbike': 12, 'Cow': 13, 'Bus': 14,
                      'Tvmonitor': 15}



    # go from data.txt to 2 vectors containing images and categories. data.txt must be in form "img path" ; "cat 1" ; "cat2 and a vector containing labels
    data_images = []
    data_classes = []
    data_attrib = []


    data_txt = open(txt_data_path, 'r')
    list_data_txt = [line.split('\n')[0] for line in data_txt.readlines()]

    parsed_list_data_txt = [elem.split(';') for elem in list_data_txt]
    # shuffled_parsed_list_data_txt = sample(parsed_list_data_txt, len(parsed_list_data_txt))

    for i in range(0, len(parsed_list_data_txt)):
        data_images.append(parsed_list_data_txt[i][0])  # x_data -> image path
        data_classes.append(ELEMENTS_CLEAN[parsed_list_data_txt[i][1]])
        data_attrib.append([i.split(',')[-1] for i in parsed_list_data_txt[i][4:]])
        #print(ELEMENTS_CLEAN[parsed_list_data_txt[i][1]])

    return data_images, data_attrib, data_classes

def parse_x_y_data_pascal_big_attrib(txt_data_path):
    ELEMENTS_CLEAN = {'Bird': 0, 'Aeroplane': 1, 'Cat': 2, 'Dog': 3, 'Sheep': 4, 'Train': 5, 'Bicycle': 6, 'Horse': 7,
                      'Bottle': 8, 'Person': 9, 'Car': 10, 'Pottedplant': 11, 'Motorbike': 12, 'Cow': 13, 'Bus': 14,
                      'Tvmonitor': 15}



    # go from data.txt to 2 vectors containing images and categories. data.txt must be in form "img path" ; "cat 1" ; "cat2 and a vector containing labels
    data_images = []
    data_classes = []
    data_attrib = []

    rectified_sub = {'Background': 0, 'Arm': 1, 'Engine': 2, 'Coach': 3, 'Tail': 4, 'Pot': 5, 'Cap': 6, 'Ear': 7,
                     'Horn': 8, 'Ebrow': 9,
                     'Nose': 10, 'Torso': 11, 'Head': 12, 'Body': 13, \
                     'Muzzle': 14, 'Beak': 15, 'Hand': 16, 'Hair': 17, 'Neck': 18, 'Foot': 19, 'Stern': 20,
                     'Artifact_Wing': 21, 'Locomotive': 22, 'License_plate': 23, \
                     'Screen': 24, 'Mirror': 25, 'Saddle': 26, 'Hoof': 27, 'Door': 28, 'Leg': 29, 'Plant': 30,
                     'Mouth': 31,
                     'Animal_Wing': 32, 'Eye': 33, 'Chain_Wheel': 34, \
                     'Bodywork': 35, 'Handlebar': 36, 'Headlight': 37, 'Wheel': 38, 'Window': 39}

    big_attrib =  {'Background': 0, 'Arm': 1, 'Engine': 2, 'Coach': 3, 'Tail': 4, 'Pot': 5, 'Ear': 6,
                     'Nose': 7, 'Torso': 8, 'Head': 9, 'Body': 10, 'Muzzle': 11, 'Hair': 12, 'Neck': 13, 'Stern': 14,'Artifact_Wing': 15, 'Locomotive': 16,  \
                     'Screen': 17, 'Door': 18, 'Leg': 19, 'Plant': 20, 'Animal_Wing': 21, 'Eye': 22, 'Chain_Wheel': 23,'Bodywork': 24, 'Headlight': 25, 'Wheel': 26, 'Window': 27}

    #big_attrib = {'Background': 0, 'Arm': 1, 'Engine': 2, 'Coach': 3, 'Pot': 4,
    #              'Nose': 5, 'Torso': 6, 'Head': 7, 'Body': 8, 'Muzzle': 9, 'Hair': 10, 'Artifact_Wing': 11,
    #              'Locomotive': 12,'Plant': 13, 'Animal_Wing': 14, 'Chain_Wheel': 15, 'Bodywork': 16, 'Headlight': 17, 'Wheel': 18}

    data_txt = open(txt_data_path, 'r')
    list_data_txt = [line.split('\n')[0] for line in data_txt.readlines()]

    parsed_list_data_txt = [elem.split(';') for elem in list_data_txt]
    # shuffled_parsed_list_data_txt = sample(parsed_list_data_txt, len(parsed_list_data_txt))

    banned_attrib = [6, 8, 9, 15, 16, 19, 23, 25, 26, 27, 31, 36]
    #banned_attrib = [3, 4, 7, 18, 20, 24, 28, 29, 33, 39, 6, 8, 9, 15, 16, 19, 23, 25, 26, 27, 31, 36]

    for i in range(0, len(parsed_list_data_txt)):
        data_images.append(parsed_list_data_txt[i][0])  # x_data -> image path
        data_classes.append(ELEMENTS_CLEAN[parsed_list_data_txt[i][1]])
        attrib_list = [i.split(',')[-1] for i in parsed_list_data_txt[i][4:]]

        #print("Attrib", attrib_list)

        big_attrib_list = [int(value) for value in attrib_list if int(value) not in banned_attrib]
        #print("Big Attrib", big_attrib_list)
        new_liste = []
        for i in big_attrib_list:
            old_name = list(rectified_sub.keys())[list(rectified_sub.values()).index(i)]
            new_index = big_attrib[str(old_name)]
            new_liste.append(new_index)


        data_attrib.append(new_liste)

        #print(ELEMENTS_CLEAN[parsed_list_data_txt[i][1]])

    return data_images, data_attrib, data_classes


def parse_x_y_data_sur_mesure(txt_data_path):
    ELEMENTS_CLEAN = {'Bird': 0, 'Aeroplane': 1, 'Cat': 2, 'Dog': 3, 'Sheep': 4, 'Train': 5, 'Bicycle': 6, 'Horse': 7,
                      'Bottle': 8, 'Person': 9, 'Car': 10, 'Pottedplant': 11, 'Motorbike': 12, 'Cow': 13, 'Bus': 14,
                      'Tvmonitor': 15}



    # go from data.txt to 2 vectors containing images and categories. data.txt must be in form "img path" ; "cat 1" ; "cat2 and a vector containing labels
    data_images = []
    data_classes = []
    data_attrib = []

    rectified_sub = {'Arm': 0, 'Engine': 1, 'Coach': 2, 'Tail': 3, 'Pot': 4, 'Cap': 5, 'Ear': 6,
                     'Horn': 7, 'Ebrow': 8,
                     'Nose': 9, 'Torso': 10, 'Head': 11, 'Body': 12, \
                     'Muzzle': 13, 'Beak': 14, 'Hand': 15, 'Hair': 16, 'Neck': 17, 'Foot': 18, 'Stern': 19,
                     'Artifact_Wing': 20, 'Locomotive': 21, 'License_plate': 22, \
                     'Screen': 23, 'Mirror': 24, 'Saddle': 25, 'Hoof': 26, 'Door': 27, 'Leg': 28, 'Plant': 29,
                     'Mouth': 30,
                     'Animal_Wing': 31, 'Eye':32, 'Chain_Wheel': 33, \
                     'Bodywork': 34, 'Handlebar': 35, 'Headlight': 36, 'Wheel': 37, 'Window': 38}

    attrib_keep = {'Arm': 0, 'Engine': 1, 'Coach': 2, 'Tail': 3, 'Pot': 4, 'Ear': 5, 'Torso': 6, 'Head': 7, 'Body': 8, \
                     'Muzzle': 9, 'Hair': 10, 'Stern': 11,
                     'Artifact_Wing': 12, 'Locomotive': 13, \
                     'Screen': 14,  'Saddle': 15, 'Leg': 16, 'Plant': 17, 'Chain_Wheel': 18, \
                     'Bodywork': 19, 'Handlebar': 20, 'Wheel': 21, 'Window': 22}


    data_txt = open(txt_data_path, 'r')
    list_data_txt = [line.split('\n')[0] for line in data_txt.readlines()]

    parsed_list_data_txt = [elem.split(';') for elem in list_data_txt]
    # shuffled_parsed_list_data_txt = sample(parsed_list_data_txt, len(parsed_list_data_txt))



    banned_attrib = [5,7,8,9,14,15,17, 18,22,24,26,27,30,31,32,36]

    for i in range(0, len(parsed_list_data_txt)):
        data_images.append(parsed_list_data_txt[i][0])  # x_data -> image path
        data_classes.append(ELEMENTS_CLEAN[parsed_list_data_txt[i][1]])
        attrib_list = [i.split(',')[-1] for i in parsed_list_data_txt[i][4:]]

        #print("Attrib", attrib_list)

        big_attrib_list = [int(value) for value in attrib_list if int(value) not in banned_attrib]
        #print("Big Attrib", big_attrib_list)
        new_liste = []
        for i in big_attrib_list:
            old_name = list(rectified_sub.keys())[list(rectified_sub.values()).index(i)]
            new_index = attrib_keep[str(old_name)]
            new_liste.append(new_index)


        data_attrib.append(new_liste)

        #print(ELEMENTS_CLEAN[parsed_list_data_txt[i][1]])

    return data_images, data_attrib, data_classes


def parse_x_y_data_4_classes_15_attribs(txt_data_path):



    # go from data.txt to 2 vectors containing images and categories. data.txt must be in form "img path" ; "cat 1" ; "cat2 and a vector containing labels
    data_images = []
    data_classes = []
    data_attrib = []

    kept_classes = ["Aeroplane", "Dog", "Car", "Pottedplant"]
    kept_classes_dict = {"Aeroplane" : 0, "Dog" : 1, "Car" : 2, "Pottedplant": 3}

    kept_attribs_dict = {'Engine': 0, 'Pot': 1, 'Ear': 2,
                'Nose': 3, 'Torso': 4, 'Head': 5, 'Body': 6, \
                'Muzzle': 7, 'Stern': 8,
                'Artifact_Wing': 9, 'Plant': 10, 'Eye': 11,
                'Bodywork': 12, 'Wheel': 13, 'Window': 14}


    rectified_sub = {'Arm': 0, 'Engine': 1, 'Coach': 2, 'Tail': 3, 'Pot': 4, 'Cap': 5, 'Ear': 6,
                     'Horn': 7, 'Ebrow': 8,
                     'Nose': 9, 'Torso': 10, 'Head': 11, 'Body': 12, \
                     'Muzzle': 13, 'Beak': 14, 'Hand': 15, 'Hair': 16, 'Neck': 17, 'Foot': 18, 'Stern': 19,
                     'Artifact_Wing': 20, 'Locomotive': 21, 'License_plate': 22, \
                     'Screen': 23, 'Mirror': 24, 'Saddle': 25, 'Hoof': 26, 'Door': 27, 'Leg': 28, 'Plant': 29,
                     'Mouth': 30,
                     'Animal_Wing': 31, 'Eye':32, 'Chain_Wheel': 33, \
                     'Bodywork': 34, 'Handlebar': 35, 'Headlight': 36, 'Wheel': 37, 'Window': 38}


    data_txt = open(txt_data_path, 'r')
    list_data_txt = [line.split('\n')[0] for line in data_txt.readlines()]

    parsed_list_data_txt = [elem.split(';') for elem in list_data_txt]


    for i in range(0, len(parsed_list_data_txt)):
        classes = parsed_list_data_txt[i][1]
        if classes in kept_classes:
            data_images.append(parsed_list_data_txt[i][0])  # x_data -> image path
            data_classes.append(kept_classes_dict[parsed_list_data_txt[i][1]])
            attrib_list = [i.split(',')[-1] for i in parsed_list_data_txt[i][4:]]

            if classes == "Aeroplane":
                banned_attrib = [3, 27, 28, 17, 24, 36, 22, 0, 2, 5, 7, 8, 14, 15, 16, 18, 21, 23, 25, 26, 30, 31, 33, 35, 37]
            else:
                banned_attrib = [3, 28, 27, 17, 24, 36, 22, 0, 2, 5, 7, 8, 14, 15, 16, 18, 21, 23, 25, 26, 30, 31, 33, 35]

            big_attrib_list = [int(value) for value in attrib_list if int(value) not in banned_attrib]

            new_liste = []
            for i in big_attrib_list:
                old_name = list(rectified_sub.keys())[list(rectified_sub.values()).index(i)]
                new_index = kept_attribs_dict[str(old_name)]
                new_liste.append(new_index)

            data_attrib.append(new_liste)


    return data_images, data_attrib, data_classes


def parse_x_y_data_4_classes(txt_data_path):



    # go from data.txt to 2 vectors containing images and categories. data.txt must be in form "img path" ; "cat 1" ; "cat2 and a vector containing labels
    data_images = []
    data_classes = []
    data_attrib = []

    kept_classes = ["Aeroplane", "Dog", "Car", "Pottedplant"]
    kept_classes_dict = {"Aeroplane" : 0, "Dog" : 1, "Car" : 2, "Pottedplant": 3}



    data_txt = open(txt_data_path, 'r')
    list_data_txt = [line.split('\n')[0] for line in data_txt.readlines()]

    parsed_list_data_txt = [elem.split(';') for elem in list_data_txt]

    for i in range(0, len(parsed_list_data_txt)):
        classes = parsed_list_data_txt[i][1]
        if classes in kept_classes:
            data_images.append(parsed_list_data_txt[i][0])  # x_data -> image path
            data_classes.append(kept_classes_dict[parsed_list_data_txt[i][1]])
            attrib_list = [i.split(',')[-1] for i in parsed_list_data_txt[i][4:]]
            data_attrib.append(attrib_list)


    return data_images, data_attrib, data_classes


def parse_x_y_data_pascal_proportion(txt_data_path):

    data_images = []
    data_classes = []
    data_attrib = []
    attrib_proportion = 0
    data_proportion = []


    data_txt = open(txt_data_path, 'r')
    list_data_txt = [line.split('\n')[0] for line in data_txt.readlines()]

    parsed_list_data_txt = [elem.split(';') for elem in list_data_txt]
    # shuffled_parsed_list_data_txt = sample(parsed_list_data_txt, len(parsed_list_data_txt))

    for i in range(0, len(parsed_list_data_txt)):
        data_images.append(parsed_list_data_txt[i][0])  # x_data -> image path
        data_classes.append(int(parsed_list_data_txt[i][2]))
        x_max = [i.split(',')[1] for i in parsed_list_data_txt[i][4:]]
        x_min = [i.split(',')[0] for i in parsed_list_data_txt[i][4:]]
        y_max = [i.split(',')[3] for i in parsed_list_data_txt[i][4:]]
        y_min = [i.split(',')[2] for i in parsed_list_data_txt[i][4:]]
        height = [i.split(',')[0] for i in parsed_list_data_txt[i][3:4]][0]
        width = [i.split(',')[1] for i in parsed_list_data_txt[i][3:4]][0]
        image_dim = int(height)*int(width)
        image_dim_list = [image_dim]*len(x_max)
        dim_x = [int(x) - int(y) for x, y in zip(x_max, x_min)]
        dim_y = [int(x) - int(y) for x, y in zip(y_max, y_min)]
        dim_attrib = [int(x) * int(y) for x, y in zip(dim_x, dim_y)]
        attrib_proportion = [int(x) / int(y) for x, y in zip(dim_attrib, image_dim_list)]

        data_proportion.append(attrib_proportion)
        data_attrib.append([i.split(',')[-1] for i in parsed_list_data_txt[i][4:]])
    return data_images, data_attrib, data_classes, data_proportion


def parse_x_y_data_proportion(txt_data_path):
    # go from data.txt to 2 vectors containing images and categories. data.txt must be in form "img path" ; "cat 1" ; "cat2 and a vector containing labels
    data_images = []
    data_categories = []
    data_label = []

    data_txt = open(txt_data_path, 'r')
    list_data_txt = [line.split('\n')[0] for line in data_txt.readlines()]
    parsed_list_data_txt = [elem.split(';') for elem in list_data_txt]

    # shuffled_parsed_list_data_txt = sample(parsed_list_data_txt, len(parsed_list_data_txt))

    for i in range(0, len(parsed_list_data_txt)):
        if int(parsed_list_data_txt[i][1]) < 100:
            data_images.append(parsed_list_data_txt[i][0])  # x_data -> image path
            if len(parsed_list_data_txt[0][3:]) >1:
                attrib_index=[]
                prop_index=[]
                for idx in range( len(parsed_list_data_txt[i][3:])):

                    attrib_index.append(int((parsed_list_data_txt[i][3:][idx][-1])))
                    prop_index.append(parsed_list_data_txt[i][3:][idx][-1])
            else:

                attrib_index = [int(float(parsed_list_data_txt[i][3:][0][-1]))  ]
                prop_index = [parsed_list_data_txt[i][3:][0][-1]  ]
            # attrib_index = [str(int(i)-1) for i in attrib_index] #just to correct a shift in the DICT
            data_categories.append(attrib_index)  # x data -> categories
            data_label.append(int(parsed_list_data_txt[i][1]))  # y data -> labels

    return data_images, data_categories, data_label



def make_hot(data):
    one_hot_list = [0] * 15  # for 39 categories for pascal voc
    for value in data:
        one_hot_list[int(value)] = 1
    return one_hot_list

def make_hot_cdg(data):
    one_hot_list = [0] * 22  # for 39 categories for pascal voc
    for value in data:
        one_hot_list[int(value)] = 1
    return one_hot_list

def make_hot_pascal(data):
    one_hot_list = [0] * 44 # for 39 categories for pascal voc
    for value in data:
        one_hot_list[int(value)] = 1
    return one_hot_list

def make_hot_pascal_clean(data):
    one_hot_list = [0] * 39 # for 39 categories for pascal voc
    for value in data:
        one_hot_list[int(value)] = 1
    return one_hot_list

def make_hot_pascal_big_attrib(data):
    one_hot_list = [0] * 27 # for 39 categories for pascal voc
    for value in data:
        one_hot_list[int(value)] = 1
    return one_hot_list

def make_hot_pascal_sur_mesure(data):
    one_hot_list = [0] * 23 # for 39 categories for pascal voc
    for value in data:
        one_hot_list[int(value)] = 1
    return one_hot_list

def make_hot_pascal_4_classes_15_attribs(data):
    one_hot_list = [0] * 15 # for 39 categories for pascal voc
    for value in data:
        one_hot_list[int(value)] = 1
    return one_hot_list

def make_hot_pascal_4_classes(data):
    one_hot_list = [0] * 39 # for 39 categories for pascal voc
    for value in data:
        one_hot_list[int(value)] = 1
    return one_hot_list

def make_hot_pascal_proportion(data, proportion):
    one_hot_list = [0] * 44  # for 39 categories for pascal voc
    proportion_list = [0] * 44
    for i in range(0, len(data)):
        one_hot_list[int(data[i])] = 1
        proportion_list[int(data[i])] += proportion[i]
    joined_list = one_hot_list + proportion_list
    return joined_list


def loss_kd(outputs, labels, teacher_outputs, temperature, alpha):
    """
    KD loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    """

    KD_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)(F.log_softmax(outputs/temperature, dim=1),teacher_outputs) * (alpha * temperature * temperature) \
              + F.cross_entropy(outputs, labels) * (1 - alpha)

    return KD_loss

def loss_kd_weighted(outputs, labels, teacher_outputs, temperature, alpha):
    """
    KD loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    """

    '''print("SIZE OUTPUTS", outputs.size())
    print("SIZE labels", labels.size())
    print("SIZE teacher_outputs", teacher_outputs.size())
    print("SIZE alpha", alpha.size())


    print("KLDIV LOSS", (nn.KLDivLoss(reduction="none", log_target=False)(F.log_softmax(outputs/temperature, dim=1),teacher_outputs)).sum(dim=1).size() )
    #print("ALPHA TEMP",  (alpha * temperature * temperature))
    print("CROSS ENTROPY", (F.cross_entropy(outputs,labels, reduction="none")).size() )
    #print("ALPHA CROSS", (1 - alpha))'''

    KD_loss = nn.KLDivLoss(reduction="none", log_target=False)(F.log_softmax(outputs/temperature, dim=1),teacher_outputs).sum(dim=1) * (alpha * temperature * temperature) \
              + F.cross_entropy(outputs,labels, reduction="none") * (1 - alpha)

    #print("KD LOSS", KD_loss)

    KD_loss = torch.mean(KD_loss)

    #print("KD LOSS MEAN", KD_loss)

    return KD_loss


def create_confusion_matrix(y_true, y_pred, classes):
    """ creates and plots a confusion matrix given two list (targets and predictions)
    :param list y_true: list of all targets (in this case integers bc. they are indices)
    :param list y_pred: list of all predictions (in this case one-hot encoded)
    :param dict classes: a dictionary of the countries with they index representation
    """

    amount_classes = len(classes)

    confusion_matrix = np.zeros((amount_classes, amount_classes))
    for idx in range(len(y_true)):
        target = y_true[idx]
        output = y_pred[idx]
        confusion_matrix[target][output] += 1

    fig, ax = plt.subplots(1)

    ax.matshow(confusion_matrix)

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            text = ax.text(j, i, confusion_matrix[i, j],
                           ha="center", va="center", color="w", fontsize=18)

    ax.set_xticks(np.arange(len(list(classes.keys()))))
    ax.set_yticks(np.arange(len(list(classes.keys()))))

    ax.set_xticklabels(list(classes.keys()))
    ax.set_yticklabels(list(classes.keys()))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    print(confusion_matrix)
    plt.show()


def count_attrib(split_x, split_y):
    zip_iterator = zip(split_y, split_x)
    list_attrib_class = [(i, j) for i, j in zip_iterator]

    dict_attrib_class = {}
    for i in split_y:
        dict_attrib_class[i] = [0] * 44

    sorted_attrib_class = sorted(list_attrib_class, key=lambda tup: tup[0])

    for i in range(0, len(sorted_attrib_class)):
        key = sorted_attrib_class[i][0]
        list_1 = dict_attrib_class[key]
        zipped_lists = zip(list_1, sorted_attrib_class[i][1])
        dict_attrib_class[key] = [x + y for (x, y) in zipped_lists]

    sorted_dict_attrib_class = sorted(dict_attrib_class.items())

    return sorted_dict_attrib_class

def count_attrib_full_dataset(train_x, train_y, val_x, val_y, test_x, test_y):

    train_attrib = count_attrib(train_x, train_y)
    val_attrib = count_attrib(val_x, val_y)
    test_attrib = count_attrib(test_x, test_y)


    for i in range(0, len(train_attrib)):
        zip_iterator = zip(train_attrib[i][1], val_attrib[i][1], test_attrib[i][1])
        total_attrib = [x + y + z for x, y, z in zip_iterator]


    return total_attrib


def apply_confidence_mask(predicted_attributes, softmax_attributes, deeplab_confidence_threshold):
    """
    Transform the raw output of deeplab into a segmentation map of confident attributes
    :param predicted_attributes, softmax_attributes:
    :return confident_pred_attributes:
    """

    # Create a GPU and a CPU copy of predicted attributes
    pred_attributes_detached = predicted_attributes.detach().max(dim=1)[1]
    pred_attributes_gpu = pred_attributes_detached.clone()

    # Calculate the max probability of attributes
    max_probs, _ = torch.max(softmax_attributes, dim=1)

    # Apply confidence mask
    mask = max_probs >= deeplab_confidence_threshold
    confident_prediction_attributes = pred_attributes_gpu * mask.float()

    return confident_prediction_attributes


def make_hot_deeplab(images, confident_pred_attributes):
    """
    Transform the segmentation map into a list of predicted attributes
    :return:
    """
    list_hot_predicted_attrib_batch = []
    for img_idx in range(len(images)):
        confident_pred_attributes_cpu = confident_pred_attributes.cpu()
        preds_attrib_numpy = np.unique(confident_pred_attributes_cpu[img_idx])
        if 0 in preds_attrib_numpy:
            corrected_preds = preds_attrib_numpy[1:] - 1
        hot_pred_class = make_hot(corrected_preds)
        list_hot_predicted_attrib_batch.append(hot_pred_class)

    return list_hot_predicted_attrib_batch

def save_ckpt(path, cur_itrs, latent_space_predictor, transparent_classifier, optimizer_lsp, optimizer_tc, scheduler_deeplab, scheduler_logreg, best_score):
    torch.save({
        "cur_itrs": cur_itrs,
        "latent_space_predictor_state": latent_space_predictor.module.state_dict(),
        "transparent_classifier_state": transparent_classifier.module.state_dict(),
        "optimizer_lsp_state": optimizer_lsp.state_dict(),
        "optimizer_tc_state": optimizer_tc.state_dict(),
        "scheduler_deeplab_state": scheduler_deeplab.state_dict(),
        "scheduler_logreg_state": scheduler_logreg.state_dict(),
        "best_score": best_score,
    }, path)
    print("Model saved as %s" % path)