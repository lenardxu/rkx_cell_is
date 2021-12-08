# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:32:19 2021

@author: SuperWZL
"""

import torch

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']
        
def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):

    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box

    s = loss_box.size(0)
    loss_box = loss_box.view(s, -1).sum(1).mean()
    # for i in sorted(dim, reverse=True):
    #   loss_box = loss_box.sum(i)
    # loss_box = loss_box.mean()
    return loss_box

def save_checkpoint(state, filename):
    torch.save(state, filename)