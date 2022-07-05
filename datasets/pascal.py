import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class Pascal(data.Dataset):
    """Pascal  Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """
    PascalClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        PascalClass('Arm',0, 0, 'construction', 0, False, False, (255,0,0)),
        PascalClass('Engine', 1, 1, 'construction', 0, False, False,(255,255,0)),
        PascalClass('Coach',2, 2, 'construction', 0, False, False, (255,255,255)),
        PascalClass('Tail',3, 3, 'construction', 0, False, False, (0,255,0)),
        PascalClass('Pot',4, 4, 'construction', 0, False, False, (0,0,255)),
        PascalClass('Cap',5, 5, 'construction', 0, False, False, (0,255,255)),
        PascalClass('Ear',6, 6, 'construction', 1, False, False, (35,255,255)),
        PascalClass('Horn',7, 7, 'construction', 1, False, False, (155,50,255)),
        PascalClass('Ebrow',8, 8, 'construction', 1, False, False, (35,150,255)),
        PascalClass('Nose',9, 9, 'construction', 1, False, False, (255,255,35)),
        PascalClass('Torso',10, 10, 'construction', 1, False, False, (0,255,35)),
        PascalClass('Head',11, 11, 'construction', 1, False, False, (20,255,35)),
        PascalClass('Body',12, 12, 'construction', 1, False, False, (40,255,35)),
        PascalClass('Muzzle',13, 13, 'construction', 1, False, False, (60,255,35)),
        PascalClass('Beak',14, 14, 'construction', 1, False, False, (80,255,35)),
        PascalClass('Hand',15, 15, 'construction', 1, False, False, (100,255,35)),
        PascalClass('Hair',16, 16, 'construction', 1, False, False, (120,255,35)),
        PascalClass('Neck',17, 17, 'construction', 1, False, False, (140,255,35)),
        PascalClass('Foot',18, 18, 'construction', 1, False, False, (160,255,35)),
        PascalClass('Stern',19, 19, 'construction', 1, False, False, (180,255,35)),
        PascalClass('Artifact_Wing',20, 20, 'construction', 1, False, False, (200,255,35)),
        PascalClass('Locomotive',21, 21, 'construction', 1, False, False, (220,255,35)),
        PascalClass('License_plate',22, 22, 'construction', 1, False, False, (240,255,35)),
        PascalClass('Screen',23, 23, 'construction', 1, False, False, (255,255,55)),
        PascalClass('Mirror',24, 24, 'construction', 1, False, False, (255,255,75)),
        PascalClass('Saddle',25, 25, 'construction', 1, False, False, (255,255,95)),
        PascalClass('Hoof',26, 26, 'construction', 1, False, False, (255,255,105)),
        PascalClass('Door',27, 27, 'construction', 1, False, False, (255,255,125)),
        PascalClass('Leg',28, 28, 'construction', 1, False, False, (255,255,145)),
        PascalClass('Plant',29, 29, 'construction', 1, False, False, (255,255,165)),
        PascalClass('Mouth',30, 30, 'construction', 1, False, False, (255,255,185)),
        PascalClass('Animal_Wing',31, 31, 'construction', 1, False, False, (255,255,205)),
        PascalClass('Eye',32, 32, 'construction', 1, False, False, (255,255,225)),
        PascalClass('Chain_Wheel',33, 33, 'construction', 1, False, False, (255,25,35)),
        PascalClass('Bodywork',34, 34, 'construction', 1, False, False, (255,50,35)),
        PascalClass('Handlebar',35, 35, 'construction', 1, False, False, (255,75,35)),
        PascalClass('Headlight',36, 36, 'construction', 1, False, False, (255,100,35)),
        PascalClass('Wheel',37, 37, 'construction', 1, False, False, (255,125,35)),
        PascalClass('Window',38, 38, 'construction', 1, False, False, (255,150,35)),
        PascalClass('unlabeled',39, 255, 'construction', 1, False, False, (0,0,0)),
    ]


    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    # Based on https://github.com/mcordts/cityscapesScripts
    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join("/home/abennetot/dataset/Pascal10/pascalpart/Deeplab/Images", split)

        self.targets_dir = os.path.join("/home/abennetot/dataset/Pascal10/pascalpart/Deeplab_masked/Images", split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        self.image_name = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for style in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, style)
            target_dir = os.path.join(self.targets_dir, style)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = file_name[:-4] + '_masked.png'

                self.targets.append(os.path.join(target_dir, target_name))

                self.image_name.append(os.path.join(img_dir, file_name))

    @classmethod
    def encode_target(cls, target):
        target = np.array(target)
        idx = target == 0
        target = target.astype(np.uint16)
        target[idx] = 256
        target = target - 1
        target = target.astype(np.uint8)
        target = Image.fromarray(target)
        return target

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 45
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        if 'Bird' in self.images[index]:
            target_class = 0
        elif 'Aeroplane' in self.images[index]:
            target_class = 1
        elif 'Cat' in self.images[index]:
            target_class = 2
        elif 'Dog' in self.images[index]:
            target_class = 3
        elif 'Sheep' in self.images[index]:
            target_class = 4
        elif 'Train' in self.images[index]:
            target_class = 5
        elif 'Bicycle' in self.images[index]:
            target_class = 6
        elif 'Horse' in self.images[index]:
            target_class = 7
        elif 'Bottle' in self.images[index]:
            target_class = 8
        elif 'Person' in self.images[index]:
            target_class = 9
        elif 'Car' in self.images[index]:
            target_class = 10
        elif 'Pottedplant' in self.images[index]:
            target_class = 11
        elif 'Motorbike' in self.images[index]:
            target_class = 12
        elif 'Cow' in self.images[index]:
            target_class = 13
        elif 'Bus' in self.images[index]:
            target_class = 14
        elif 'Tvmonitor' in self.images[index]:
            target_class = 15


        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        name = self.images[index]

        cond = True
        while cond == True:
            if self.transform:
                image_final, target_final = self.transform(image, target)
                target_final_tmp = target_final.numpy()

            if len(np.unique(target_final_tmp)) > 1:
                cond = False
                return name, image_final, target_final, target_class

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)