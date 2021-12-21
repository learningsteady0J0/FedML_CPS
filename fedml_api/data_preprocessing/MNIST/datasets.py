import logging
import numpy as np
import torch.utils.data as data
from PIL import Image

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class MNIST_truncated(data.Dataset):

    def __init__(self, data,target, dataidxs=None, transform=None, target_transform=None):
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.target = self.__build_truncated_dataset__(data, target)

    def __build_truncated_dataset__(self, data,target):
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
