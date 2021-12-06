# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 22:22:35 2021

@author: SuperWZL
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.config import cfg

class SSB(nn.Module):
    def __init__(self, second_FM_Channel):
        self.second_FM_Channel = second_FM_Channel
        
        self.Conv_FP = nn.Conv2d(in_channels, out_channels, 1)
        
        self.Conv_Mid = nn.Sequential(
            nn.Conv2d(self.second_FM_Channel, self.second_FM_Channel, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.second_FM_Channel, self.second_FM_Channel, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.second_FM_Channel, self.second_FM_Channel, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.second_FM_Channel, self.second_FM_Channel, 3, padding=1, stride=1),
            nn.ReLU()
            )
        
    def forward(self, p2, p3, p4, p5, p6):
        def Conv(p):
            
        
        