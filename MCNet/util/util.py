"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os



#####
from torchvision import utils

def unnormalize(img_tensor, unnormalize_STD=[0.5], unnormalize_MEAN=[0.5]):

    #img_tensor = img_tensor.transpose(1, 3)
    MEAN_tensor = torch.Tensor(unnormalize_STD).type(img_tensor.type())
    STD_tensor = torch.Tensor(unnormalize_MEAN).type(img_tensor.type())
    #print(img_tensor.type(), MEAN_tensor.type(), STD_tensor.type())
    img_tensor = img_tensor * STD_tensor + MEAN_tensor
    #img_tensor = img_tensor.transpose(1, 3)

    return img_tensor



def save_current_images(results, img_path, unnormalize_STD=[0.5], unnormalize_MEAN=[0.5]):
    '''
    :param results (OrderedDict)  : -- an ordered dictionary that stores (name, tensor ) pairs
    :param img_path (str): -- the string is used to save image paths

    Notice: (1) one-channel image
            (2) the value of unnormalize_STD and unnormalize_MEAN comes from  normalize_params of dataset

    :return:
    '''
    tensor_list =[]
    for label, im_data in results.items():
        if label != 'mask_M' and label !='fake_B_gradient' and label !='real_B_gradient' :
            tensor_list.append(unnormalize(im_data,  unnormalize_STD = unnormalize_STD, unnormalize_MEAN = unnormalize_MEAN))
        else:
            #print(label)
            tensor_list.append(maxmin_normalize_tensor(im_data))

    if len(tensor_list) == 8:
        tensorA = tensor_list[0]*tensor_list[1] # tensor_list[0] is not zero in mask region Because of unnormalize
        group_tensor = torch.cat((tensorA.data, tensor_list[1].data, tensor_list[2].data,
                                  tensor_list[3].data, tensor_list[4].data, tensor_list[5].data, tensor_list[6].data, tensor_list[7].data), -1)
        utils.save_image(group_tensor, img_path, nrow=1, padding=0)
        #half_range_save_image(group_tensor, img_path, nrow=1, padding=0)
        # convert RGB to L and save it
        imgL = Image.open(img_path).convert('L')
        imgL.save(img_path)
    else:
        print('The number of Input tensor is wrong !')

def maxmin_normalize_tensor(in_tensor):
    tensor_new = in_tensor.clone()
    min=float(tensor_new.min())
    max =float(tensor_new.max())
    tensor_new.clamp_(min=min, max=max)
    tensor_new.add_(-min).div_(max - min + 1e-5)
    return tensor_new


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

'''
if __name__ == "__main__":
    in_tensor=torch.tensor([[-0.5,0.5,1.5],[-0.5,1.0,0.9]])
    a=maxmin_normalize_tensor(in_tensor)
    print(in_tensor)
    print(a)
'''