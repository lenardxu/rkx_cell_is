# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:16:52 2021

@author: SuperWZL
"""

import numpy as np
import torch
import cv2

class DateType:
    def numpy2torch(array):
        """ Converts 3D numpy HWC ndarray to 3D PyTorch CHW tensor."""
        if array.ndim == 3:
            array = np.transpose(array, (2, 0, 1))
        elif array.ndim == 2:
            array = np.expand_dims(array, axis=0)
        return torch.from_numpy(array)


    def torch2numpy(tensor):
        """ Convert 3D PyTorch CHW tensor to 3D numpy HWC ndarray."""
        assert (tensor.dim() == 3)
        return np.transpose(tensor.numpy(), (1, 2, 0))
"""
class ImgSize:
    def changeSize(img, desSize = (512, 512, 3)):
        new_img = cv2.resize(img, desSize)
        return new_img
"""