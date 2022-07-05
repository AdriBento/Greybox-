import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class CDGai(data.Dataset):
    """
    XAI CDG Dataset.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CDGai = namedtuple('CDGai', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CDGai('Unlabeled', 0, 0, 'unlabeled', 0, False, False, (0,0,0)),
        CDGai('Building', 1, 1, 'Building', 1, False, False, (70,70,70)),
        CDGai('Fence', 2, 2, 'Fence', 2, False, False, (100,40,40)),
        CDGai('Other', 3, 3, 'Other', 3, False, False, (55, 90, 80)),
        CDGai('Pedestrian', 4, 4, 'Pedestrian', 4, False, False, (220, 20, 60)),
        CDGai('Pole', 5, 5, 'Pole', 5, False, False, (153, 153, 153)),
        CDGai('RoadLine', 6, 6, 'RoadLine', 6, False, False, (157, 234, 50)),
        CDGai('Road', 7, 7, 'Road', 7, False, False, (128, 64, 128)),
        CDGai('SideWalk', 8, 8, 'SideWalk', 8, False, False, (244, 35, 232)),
        CDGai('Vegetation', 9, 9, 'Vegetation', 9, False, False, (107, 142, 35)),
        CDGai('Vehicles', 10, 10, 'Vehicles', 10, False, False, (0, 0, 142)),
        CDGai('Wall', 11, 11, 'Wall', 11, False, False, (102, 102, 156)),
        CDGai('TrafficSign', 12, 12, 'TrafficSign', 12, False, False, (220, 220, 0)),
        CDGai('Sky', 13, 13, 'Sky', 13, False, False, (70, 130, 180)),
        CDGai('Ground', 14, 14, 'Ground', 14, False, False, (81, 0, 81)),
        CDGai('Bridge', 15, 15, 'Bridge', 15, False, False, (150, 100, 100)),
        CDGai('RailTrack', 16, 16, 'RailTrack', 16, False, False, (230, 150, 140)),
        CDGai('GuardRail', 17, 17, 'GuardRail', 17, False, False, (180, 165, 180)),
        CDGai('TrafficLight', 18, 18, 'TrafficLight', 18, False, False, (250, 170, 30)),
        CDGai('Static', 19, 19, 'Static', 19, False, False, (110, 190, 160)),
        CDGai('Dynamic', 20, 20, 'Dynamic', 20, False, False, (170, 120, 50)),
        CDGai('Water', 21, 21, 'Water', 21, False, False, (45, 60, 150)),
        CDGai('Terrain', 22, 2, 'Terrain', 22, False, False, (145, 170, 100))
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])


    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join("/home/abennetot/dataset/CDGai/cdgai", split)

        self.targets_dir = os.path.join("/home/abennetot/dataset/CDGai/cdgai_rss/backup", split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for classe in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, classe)
            target_dir = os.path.join(self.targets_dir, classe)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = file_name
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
        # return cls.id_to_train_id[np.array(target)]

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

        if 'C0' in self.images[index]: target_class = 0
        elif 'C1' in self.images[index]: target_class = 1
        elif 'C2' in self.images[index]: target_class = 2
        elif 'C3' in self.images[index]: target_class = 3
        elif 'C4' in self.images[index]: target_class = 4
        elif 'C5' in self.images[index]: target_class = 5
        elif 'C6' in self.images[index]: target_class = 6
        elif 'C7' in self.images[index]: target_class = 7
        elif 'C8' in self.images[index]: target_class = 8
        elif 'C9' in self.images[index]: target_class = 9
        elif 'C10' in self.images[index]: target_class = 10
        elif 'C11' in self.images[index]: target_class = 11
        elif 'C12' in self.images[index]: target_class = 12

        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        cond = True
        while cond == True:
            if self.transform:
                image_final, target_final = self.transform(image, target)
                target_final_tmp = target_final.numpy()
                #
                # cond=False #quick fix remove when binary masks are ready

            if len(np.unique(target_final_tmp)) > 1:
                cond = False

                return image_final, target_final, target_class


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
