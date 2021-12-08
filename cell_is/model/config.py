# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:20:40 2021

@author: SuperWZL
"""


import os
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.NCLASSES = 2


__C.TRAIN = edict()

# Original image size
__C.TRAIN.WIDTH = 520
__C.TRAIN.HEIGTH = 704
__C.TRAIN.SHAPE = (520, 704)

# Learning rate of training
__C.TRAIN.LEARNING_RATE = 0.001

# Cell-Type
__C.TRAIN.CELL_TYPE = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'RatC6', 'SHSY5Y', 'SkBr3', 'SKOV3']

__C.PATH= edict()

# Root path of dataset
__C.PATH.ROOT = 'D:/Kaggle/cell/sartorius-cell-instance-segmentation/'

# Path of current dataset
__C.PATH.TRAIN = os.path.join(__C.PATH.ROOT, 'train')
__C.PATH.TEST = os.path.join(__C.PATH.ROOT, 'test')
__C.PATH.TRAIN_CSV = os.path.join(__C.PATH.ROOT, 'train.csv')

# Root path of LiveCell-Dataset
__C.PATH.LIVE_ROOT = os.path.join(__C.PATH.ROOT, 'LIVECell_dataset_2021')

# Path of LiveCell-Dataset
__C.PATH.LIVE_TRAIN_VAL_IMAGES = os.path.join(__C.PATH.LIVE_ROOT, 'livecell_train_val_images')
__C.PATH.LIVE_TEST_IMAGES = os.path.join(__C.PATH.LIVE_ROOT, 'livecell_test_images')
__C.PATH.LIVE_TRAIN_JSON = os.path.join(__C.PATH.LIVE_ROOT, 'annotations/LIVECell/livecell_coco_train.json')
__C.PATH.LIVE_VAL_JSON = os.path.join(__C.PATH.LIVE_ROOT, 'annotations/LIVECell/livecell_coco_val.json')
__C.PATH.LIVE_TEST_JSON = os.path.join(__C.PATH.LIVE_ROOT, 'annotations/LIVECell/livecell_coco_test.json')

# Parameters about anchor
__C.ANCHOR_RATIOS = [0.5, 1, 2]
__C.ANCHOR_SCALES = [8, 16, 32]
__C.STRIDE = [16, ]




