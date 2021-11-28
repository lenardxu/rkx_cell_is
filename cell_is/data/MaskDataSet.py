# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 23:32:26 2021

@author: SuperWZL
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.config import cfg

TRAIN_CSV = pd.read_csv(cfg.TRAIN_CSV_PATH)

class MaskDataSet:
    """
        create mask-imgs for training-imgs
    """
    def _creat_one_mask(self, mask_code_lst, shape = cfg.TRAIN.SHAPE):
        """
        Return the corresponding mask-image through the given annotation of one image
            parameters:
                mask_code_lst (list): list of annotation of one image
            return:
                mask (ndaarray): shape = shape(520, 704)
        """
        mask = np.zeros(shape = (shape[0] * shape[1]))
        for mask_code in mask_code_lst:
            mask_code = mask_code.split()
            starts = list(map(int, mask_code[::2]))
            lens = list(map(int, mask_code[1::2]))
            for _start, _len in zip(starts, lens):
                mask[_start - 1: _start + _len] = 1
        mask = mask.reshape(shape)
        return mask
    
    def creat_mask(self, df = TRAIN_CSV):
        """
        Return all Mask-Image of training dataset
            parameter:
                df: csv-File of training data
            return:
                mask (dict): The dictionary contains all mask-images, {key : value} -> {img_id : mask}
        """
        mask = {}
        ids = df['id']
        mask_code_lst = []
        for number, img_id in enumerate(ids):
            if(img_id not in mask.keys()):
                if number == 0:
                    mask[img_id] = None
                else:
                    tmp_mask = self._creat_one_mask(mask_code_lst)
                    mask[df.iloc[number-1]['id']] = tmp_mask
                    mask[img_id] = None
                    mask_code_lst = []
            mask_code_lst.append(df.iloc[number]['annotation'])
        return mask