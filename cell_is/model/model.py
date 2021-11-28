# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 23:48:52 2021

@author: SuperWZL
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class NN_Model(nn.Module):
    def __init__(self):
        super(NN_Model).__init__();
        
        
