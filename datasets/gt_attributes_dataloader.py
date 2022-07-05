import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from utils import make_x_y_data

class Monumai_Attributes(data.Dataset):

    def __init__(self, root, transform=None):


        self.images_path, self.attributes, self.classes =  make_x_y_data(root)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image_path = self.images_path[index]
        attributes = self.attributes[index]

        classes = self.classes[index]

        return attributes, classes

    def __len__(self):
        return len(self.images_path)