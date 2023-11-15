import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import v2
import torchvision.transforms.functional as TF

def local_enhancement(img, filter_size=65, sigma=40,convert2pil=True):
    #image = np.array(img)
    image_blur = cv2.GaussianBlur(img,(filter_size,filter_size),sigma)
    # new_image = cv2.subtract(img,image_blur).astype('float32') # WRONG, the result is not stored in float32 directly
    new_image = cv2.subtract(img,image_blur, dtype=cv2.CV_32F)
    out = cv2.normalize(new_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if convert2pil:
        out = TF.to_tensor(Image.fromarray(out.astype('uint8'), 'RGB'))

    return out