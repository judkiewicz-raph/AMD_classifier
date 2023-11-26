import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
import glob
import json

import torch
import torchvision.transforms.functional as TF

from utils.utilities import get_label_from_filename

def read_data(dir_path, admissible_labels=[], single_file=False, resize=None):
    """
    Read the data, apply transformations (remove characte in upper left corner and local contrast enhancement) and returns list of images and list of labels.

    Args:
        dir_path (str): The directory path where the data is located.
        admissible_labels (list, optional): A list of admissible labels. Only images with labels in this list will be included. If not provided, all labels are considered admissible. Default is an empty list.
        single_file (bool, optional): Whether the data consists of a single file. If True, dir_path should be the path to the image file. Default is False.
        resize (function, optional): A function to resize the images. If provided, the images will be resized using this function. Default is None.

    Returns:
        tuple: A tuple containing a list of processed images and a list of corresponding labels.

    """

    if single_file:
        img_path = dir_path
        label = get_label_from_filename(img_path)
        if label in admissible_labels or not admissible_labels:
            img = plt.imread(img_path)
            img = remove_character(img)
            img_enh = local_enhancement(img,65,40,True)
            if resize:
                img_enh = resize(img_enh)
        return img_enh, label

    images = []
    labels = []

    for img_path in glob.glob(os.path.join(dir_path,"*.png")):
        
        label = get_label_from_filename(img_path)

        if label in admissible_labels or not admissible_labels:
            labels.append(label)
            img = plt.imread(img_path)
            img = remove_character(img)
            img_enh = local_enhancement(img,65,40,True)
            if resize:
                img_enh = resize(img_enh)

            images.append(img_enh)

    images = torch.stack(images)

    return images, labels

def remove_character(image, size_x=120, size_y=70):
    """
    Remove the white character showing in certain images of diseased eye.

    Args:
        image (ndarray): The input image as a NumPy array.
        size_x (int, optional): The width of the region to remove the character from. Default is 120.
        size_y (int, optional): The height of the region to remove the character from. Default is 70.

    Returns:
        ndarray: The image with the character removed.
    """
    image[:size_x,:size_y] = np.zeros_like(image[:size_x,:size_y])
    return image

def local_enhancement(img, filter_size=65, sigma=40,convert2tensor=True):
    """
    Applies local enhancement to an image using gaussian blur, and returns the image either as a PIL.Image, or a torch.tensor 
       
    Args:
        img (ndarray): The input image as a NumPy array.
        filter_size (int, optional): The size of the Gaussian filter. Default is 65.
        sigma (int, optional): The standard deviation of the Gaussian filter. Default is 40.
        convert2tensor (bool, optional): Whether to convert the output image to a torch.Tensor or return it as a ndarray. Default is True.

    Returns:
        ndarray or torch.Tensor: The enhanced image. If convert2tensor is True, it is returned as a torch.Tensor. Otherwise, it is returned as a ndarray.
    """

    #image = np.array(img)
    image_blur = cv2.GaussianBlur(img,(filter_size,filter_size),sigma)
    # new_image = cv2.subtract(img,image_blur).astype('float32') # WRONG, the result is not stored in float32 directly
    new_image = cv2.subtract(img,image_blur, dtype=cv2.CV_32F)
    out = cv2.normalize(new_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if convert2tensor:
        out = TF.to_tensor(Image.fromarray(out.astype('uint8'), 'RGB'))

    return out

def Normalize(images, path_load=None, path_save=None):
    """
    Normalize a set of images by subtracting the mean and dividing by the standard deviation.

    Args:
        images (ndarray): The input images as a NumPy array.
        path_load (str, optional): The path to a JSON file containing normalization configuration. If provided, the mean and standard deviation will be loaded from this file. Default is None.
        path_save (str, optional): The path to save the normalization configuration as a JSON file. If provided, the mean and standard deviation will be saved to this file. Default is None.

    Returns:
        ndarray: The normalized images.
    """

    if path_load:
        with open(path_load, 'rb') as f:
            normalize_config = json.load(f)
            mean, std = torch.tensor(normalize_config['mean']), normalize_config['std']
    else:
        mean = images.mean(axis=0)
        std = images.std(axis=0).mean()

    images = (images-mean)/std

    if path_save:
        normalize_config = {'mean': mean.numpy().tolist(), 'std':std.item()}
        with open(path_save,'w') as f:
            json.dump(normalize_config,f)
    
    return images
