"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


# def get_flip(opt):
#
#     if opt.flip:
#         flip = random.random() > 0.5
#     else:
#         flip = False
#
#     return flip


def get_rotate_param(option_rotate=True):

    if option_rotate:
        rotate_flag = random.sample(range(0,4),1)[0] # 0: no roate, 1:rotate 90, 2:rotate 180, 3:rotate 270
                                                     # random.sample(range(0,4),1) is a list
    else:
        rotate_flag = 0

    return rotate_flag



def img_rotate(input_img, rotate_flag):
    #print(rotate_flag)
    if rotate_flag ==0:
        result_img = input_img
        return result_img
    elif rotate_flag ==1:
        result_img = input_img.transpose(Image.ROTATE_90)
        return result_img
    elif rotate_flag ==2:
        result_img = input_img.transpose(Image.ROTATE_180)
        return result_img
    elif rotate_flag ==3:
        result_img = input_img.transpose(Image.ROTATE_270)
        return result_img
    else:
        print('rotate_flag is wrong')




def get_flip_param(option_flip=True):

    if option_flip:
        flip = random.random() > 0.5
    else:
        flip = False

    return flip


def img_flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_transform(grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if convert:
        ###transform_list += [transforms.ToTensor(),
        ###                   transforms.Normalize((0.5, 0.5, 0.5),
        ###                                        (0.5, 0.5, 0.5))]
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize(mean=(0.5,), std=(0.5,))]
        ###  type((0.5,)) is tuple, but type((0.5)) is int
    else:
        transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

# def get_transform_test(grayscale=False, convert=True):
#     transform_list = []
#     if grayscale:
#         transform_list.append(transforms.Grayscale(1))
#
#     if convert:
#         ###transform_list += [transforms.ToTensor(),
#         ###                   transforms.Normalize((0.5, 0.5, 0.5),
#         ###                                        (0.5, 0.5, 0.5))]
#         transform_list += [transforms.ToTensor(),
#                            transforms.Normalize(mean=(0.5,), std=(0.5,))]
#         ###  type((0.5,)) is tuple, but type((0.5)) is int
#     else:
#         transform_list += [transforms.ToTensor()]
#
#     return transforms.Compose(transform_list)


def random_crop_transform(loadSize=350, cropSize=256, option_flip=True, option_rotate=True, img_list=[]):

    i,j,th,tw = get_crop_params(input_size=(350, 350), output_size=(256, 256))
    flip_param = get_flip_param(option_flip)
    rotate_param = get_rotate_param(option_rotate)
    #print(i,j,th,tw, flip_param)
    cropped_img_list=[]
    for img in img_list:
        cropped_img_list.append(img_flip((img_rotate(F.crop(img, i, j, th, tw),rotate_param)), flip_param))
    #F.crop(img, i, j, h, w)
    return cropped_img_list



def get_crop_params(input_size=(350,350), output_size=(256,256)):
    """Get parameters for ``crop`` for a random crop.

    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    w, h = input_size
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(20, h - th - 20)  ###
    j = random.randint(20, w - tw - 20)  ###
    return i, j, th, tw

# def get_transform_train(grayscale=False, loadSize= 350, cropSize=256, convert=True):
#     transform_list = []
#     if grayscale:
#         transform_list.append(transforms.Grayscale(1))
#
#     if convert:
#         ###transform_list += [transforms.ToTensor(),
#         ###                   transforms.Normalize((0.5, 0.5, 0.5),
#         ###                                        (0.5, 0.5, 0.5))]
#         transform_list += [transforms.Resize(size=loadSize, interpolation=Image.BICUBIC),
#                            transforms.RandomCrop(size=cropSize),
#                            transforms.RandomHorizontalFlip(p=0.5),
#                            transforms.ToTensor(),
#                            transforms.Normalize(mean=(0.5,), std=(0.5,))]
#         ###  type((0.5,)) is tuple, but type((0.5)) is int
#     else:
#         transform_list += [transforms.ToTensor()]
#
#     return transforms.Compose(transform_list)


# def ImageTransform(loadSize, cropSize):
#     return Compose([
#         Resize(size=loadSize, interpolation=Image.BICUBIC),
#         RandomCrop(size=cropSize),
#         RandomHorizontalFlip(p=0.5),
#         ToTensor(),
#     ])





'''
def get_transform(opt, flip=False,  convert=True):


    transform_list = []
    if opt.preprocess =='resize':
        transform_list.append(transforms.Resize((opt.img_height, opt.img_width)))

    if flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, flip)))

    if convert:
        ###transform_list += [transforms.ToTensor(),
        ###                   transforms.Normalize((0.5, 0.5, 0.5),
        ###                                        (0.5, 0.5, 0.5))]
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize(mean=(0.5,), std=(0.5,))]
        ###  type((0.5,)) is tuple, but type((0.5)) is int
    else:
        transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)
'''



def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


