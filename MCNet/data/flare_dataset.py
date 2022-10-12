import os.path
import random
from data.base_dataset import BaseDataset, get_transform, random_crop_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from random import randint



class FlareDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dir_LSM = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.LSM_paths = sorted(make_dataset(self.dir_LSM, opt.max_dataset_size))  # get image paths

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        L - - an image of long  exptime
        S - - an image of short exptime
        M - - an image of mask
        Returns a dictionary that contains A, B, M, Refer,paths
            A (tensor) - - an image of long  exptime
            B (tensor) - - an image of short exptime
            M (tensor) - - an image of mask
            Refer (tensor) - - an image L or s
            paths (str) - - image paths (L,S,M)

        """
        # read a image given a random integer index
        LSM_path = self.LSM_paths[index]
        ###AB = Image.open(AB_path).convert('RGB')
        LSM = Image.open(LSM_path).convert('L')
        # split LSM image into L, S and M
        w, h = LSM.size
        w0 = int(w / 3)
        L = LSM.crop((0, 0, w0, h))
        S = LSM.crop((w0, 0, w0*2, h))
        M = LSM.crop((w0*2, 0, w0*3, h))



        # apply the same transform to L, S and M
        #flip_param = get_flip(self.opt)
        L_transform = get_transform(grayscale=(self.input_nc == 1), convert=True)
        S_transform = get_transform(grayscale=(self.input_nc == 1), convert=True)
        M_transform = get_transform(grayscale=(self.input_nc == 1), convert=False)



        # ToTensor(object) of self.mask_transform(mask.convert('L'))
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]

        ###A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        ###B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        if self.opt.isTrain == True and self.opt.data_type == 'short':
            LSM_list = random_crop_transform(loadSize=350, cropSize=256, option_flip=True, option_rotate=True,img_list=[L,S,M])
            L = LSM_list[0]
            S = LSM_list[1]
            M = LSM_list[2]

            M = M_transform(M)
            B = S_transform(S)
            A = B * M
            R = L_transform(L) # R--->Refer

        if self.opt.isTrain == False and self.opt.data_type == 'long':
            M = M_transform(M)
            B = L_transform(L)
            A = B * M
            #B = M + (1.0-M)
            R = S_transform(S) # R--->Refer

        if self.opt.isTrain == False and self.opt.data_type == 'short':
            M = M_transform(M)
            B = S_transform(S)
            A = B * M
            R = L_transform(L) # R--->Refer


        return {'A': A, 'B': B, 'M': M, 'R': R, 'paths': LSM_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.LSM_paths)
