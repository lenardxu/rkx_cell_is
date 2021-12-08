# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 22:22:35 2021

@author: SuperWZL
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.config import cfg

class SSB(nn.Module):
    def __init__(self, second_FM_Channel):
        self.second_FM_Channel = second_FM_Channel
        
        self.Conv_FP = nn.Conv2d(self.second_FM_Channel, self.second_FM_Channel, 1, 8)
        
        self.Block_Mid = nn.Sequential(
            nn.Conv2d(self.second_FM_Channel, self.second_FM_Channel, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.second_FM_Channel, self.second_FM_Channel, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.second_FM_Channel, self.second_FM_Channel, 3, padding=1, stride=1),
            nn.ReLU()
            )
        
    def forward(self, p2, p3, p4, p5, p6, nclasses = cfg.NCLASSES):
        size_FeatureMap = p3.size();
        p3 = nn.ReLU(self.Conv_FP(p3))
        
        p2 = nn.ReLU(self.Conv_FP(p2))
        p2 = cv2.resize(p2, size_FeatureMap)
        
        p4 = nn.ReLU(self.Conv_FP(p4))
        p4 = cv2.resize(p4, size_FeatureMap)
        
        p5 = nn.ReLU(self.Conv_FP(p5))
        p5 = cv2.resize(p5, size_FeatureMap)
        
        p6 = nn.ReLU(self.Conv_FP(p6))
        p6 = cv2.resize(p6, size_FeatureMap)
        
        P = p2 + p3 + p4 + p5 + p6
        
        out = self.Block_Mid(P)
        
        pre_semantic = nn.ReLU(nn.Conv2d(self.second_FM_Channel, self.second_FM_Channel // 4, 3, padding=1, stride=1))
        semantic = nn.Conv2d(self.second_FM_Channel//4, nclasses, 1)
        
        
        return semantic
        
        
            
        
        