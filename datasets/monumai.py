import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class Monumai(data.Dataset):
    """Monumai  Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    MonumaiClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        MonumaiClass('arco-herradura',          0, 0, 'construction', 0, False, False, (255,0,0)),
        MonumaiClass('dintel-adovelado', 1, 1, 'construction', 0, False, False,(255,255,0)),
        MonumaiClass('arco-lobulado',           2, 2, 'construction', 0, False, False, (255,255,255)),
        MonumaiClass('arco-medio-punto',               3, 3, 'construction', 0, False, False, (0,255,0)),
        MonumaiClass('arco-apuntado',              4, 4, 'construction', 0, False, False, (0,0,255)),
        MonumaiClass('vano-adintelado',               5, 5, 'construction', 0, False, False, (0,255,255)),
        MonumaiClass('fronton',                 6, 6, 'construction', 1, False, False, (35,255,255)),
        MonumaiClass('arco-conopial',             7, 7, 'construction', 1, False, False, (155,50,255)),
        MonumaiClass('arco-trilobulado',              8, 8, 'construction', 1, False, False, (35,150,255)),
        MonumaiClass('serliana',           9, 9, 'construction', 1, False, False, (255,255,35)),
        MonumaiClass('ojo-de-buey',             10, 10, 'construction', 2, False, False, (255,50,35)),
        MonumaiClass('fronton-curvo',                 11, 11, 'construction', 2, False, False,(255,35,50)),
        MonumaiClass('fronton-partido',                12, 12, 'construction', 2, False, False, (255,35,50)),
        MonumaiClass('columna-salomonica',           13, 13, 'construction', 2, False, False, (255,35,150)),
        MonumaiClass('pinaculo-gotico',               14, 14, 'construction', 2, False, False, (255,150,150)),
        MonumaiClass('unlabeled', 15, 255, 'void', 0, False, True, (0, 0, 0))
    ]


    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join("/home/abennetot/dataset/OD-MonuMAI/MonuMAI_dataset/FlowFromDirectory", split)

        self.targets_dir = os.path.join("/home/abennetot/dataset/OD-MonuMAI/MonuMAI_dataset", split)
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
                target_name = file_name[:-4] +'_binary_mask.png'
                self.targets.append(os.path.join(target_dir, target_name))

    @classmethod
    def encode_target(cls, target):
        target=np.array(target)
        idx =target==0
        target=target.astype(np.uint16)
        target[idx]=256
        target=target-1
        target = target.astype(np.uint8)
        target=Image.fromarray(target)
        return target

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 15
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        if 'Baroque' in self.images[index]: target_class = 3
        elif 'Gothic' in self.images[index]: target_class = 1
        elif 'Hispanic' in self.images[index]: target_class = 0
        elif 'Renaissance' in self.images[index]: target_class = 2

        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        name = self.images[index]

        cond=True
        while cond ==True:
            if self.transform:
                image_final, target_final = self.transform(image, target)
                target_final_tmp=target_final.numpy()


            if len(np.unique(target_final_tmp)) >1:
                cond = False
                return name, image_final, target_final,target_class

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