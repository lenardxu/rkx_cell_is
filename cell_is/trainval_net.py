# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:05:58 2021

@author: SuperWZL
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import argparse
import time
import os

from model.utils import adjust_learning_rate, save_checkpoint
from model.config import cfg

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--isAnchor_based', default=True, dest='isAnchor_based',
                        help='choose anchor-based or anchor-free')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='detnet59, etc',
                        default='detnet59', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="/srv/share/jyang375/models", )
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--lscale', dest='lscale',
                        help='whether use large scale',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=True, type=bool)
    parser.add_argument('--cascade', help='whether use cascade structure', action='store_true')

    args = parser.parse_args()
    return args

def _print(str, logger=None):
    print(str)
    if logger is None:
        return
    logger.info(str)

if __name__ == '__main__':
    args = parse_args()
    """
        ......(dataloader)
    """
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    
    if args.cuda:
        cfg.CUDA = True
        
    if args.isAnchor_based:
        if args.net == 'CBNetV2+HTC':
            model = CBNet_HTC(...)
        else:
            print("network is not defined")
            #pdb.set_trace() #debug?
    else:# need the information of the other group
        if args.net == '':
            model = ...
        else:
            print("network is not defined")
            #pdb.set_trace() #debug?
            
    model.create_architecture()
    
    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value], 'lr':lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    
    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    
    
    if args.resume:
        load_name = os.path.join(output_dir,
                                 'model_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        _print("loading checkpoint %s" % (load_name), )
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        _print("loaded checkpoint %s" % (load_name), )
    
    if args.mGPUs:
        model = nn.DataParallel(model)

    if args.cuda:
        model.cuda()
        
    iters_per_epoch = int(train_size / args.batch_size)
    
    for epoch in range(args.start_epoch, args.max_epochs):
        model.train()
        loss_temp = 0
        start = time.time
        
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
            
        data_iter = iter(dataloader)
        
        for step in range(iters_per_epoch):
            data = data_iter.next()
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])
            
            model.zero_grad()
            
            ##########################################################################################################
            # loss-function:                                                                                         #
            # L = \sum_1^t (alpha_t(L_bbox_t + L_mask_t)) + beta*L_seg: loss of the bounding box predictions         #
            # L_bbox_t = L_cls + L_reg, same as Cascade R-CNN                                                        #
            # L_mask_t: the loss of mask prediction at stage t, same as Mask-RCNN                                    #
            # L_seg: the semantic segmentation loss in the form of cross entropy                                     #
            # alpha = [1, 0.5, 0.25]                                                                                 #
            # beta = 1                                                                                               #
            # T = 3                                                                                                  #
            ##########################################################################################################
            T = 3
            beta = 1
            alpha = [1, 0.5, 0.25]
            loss_cls_lst, loss_reg_lst, loss_mask_lst, loss_seg, roi_labels = model(im_data, im_info, gt_boxes, num_boxes)
            loss_sum = 0
            for i in range(T):
                loss_sum += (alpha[i] * (loss_cls_lst[i].mean() + loss_reg_lst[i].mean() + loss_mask_lst[i].mean()))
            loss = loss_sum + beta * loss_seg.mean()
            
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    loss_cls_1 = loss_cls_lst[0].mean().data[0]
                    loss_cls_2 = loss_cls_lst[1].mean().data[0]
                    loss_cls_3 = loss_cls_lst[2].mean().data[0]
                    
                    loss_reg_1 = loss_reg_lst[0].mean().data[0]
                    loss_reg_2 = loss_reg_lst[1].mean().data[0]
                    loss_reg_3 = loss_reg_lst[2].mean().data[0]
                    
                    loss_mask_1 = loss_mask_lst[0].mean().data[0]
                    loss_mask_2 = loss_mask_lst[1].mean().data[0]
                    loss_mask_3 = loss_mask_lst[2].mean().data[0]
                    
                    loss_seg_ = loss_seg.mean().data[0]

                    fg_cnt = torch.sum(roi_labels.data.ne(0))
                    bg_cnt = roi_labels.data.numel() - fg_cnt
                else:
                    loss_cls_1 = loss_cls_lst[0].data[0]
                    loss_cls_2 = loss_cls_lst[1].data[0]
                    loss_cls_3 = loss_cls_lst[2].data[0]
                    
                    loss_reg_1 = loss_reg_lst[0].data[0]
                    loss_reg_2 = loss_reg_lst[1].data[0]
                    loss_reg_3 = loss_reg_lst[2].data[0]
                    
                    loss_mask_1 = loss_mask_lst[0].data[0]
                    loss_mask_2 = loss_mask_lst[1].data[0]
                    loss_mask_3 = loss_mask_lst[2].data[0]
                    
                    loss_seg_ = loss_seg.data[0]

                    fg_cnt = torch.sum(roi_labels.data.ne(0))
                    bg_cnt = roi_labels.data.numel() - fg_cnt

                _print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                       % (args.session, epoch, step, iters_per_epoch, loss_temp, lr), )
                _print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start), )
                '''
                if args.cascade:
                    _print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, rcnn_cls_2nd: %.4f, "
                           "rcnn_box_2nd %.4f, rcnn_cls_3rd: %.4f, rcnn_box_3rd %.4f" % (loss_rpn_cls, loss_rpn_box,
                        loss_cls_1, loss_reg_1, loss_rcnn_cls_2nd, loss_rcnn_box_2nd, loss_rcnn_cls_3rd, loss_rcnn_box_3rd), )
                else:
                    _print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                            % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box), )

                if args.use_tfboard:
                    if args.cascade:
                        scalars = [loss_temp, loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_rcnn_cls_2nd, loss_rcnn_box_2nd, loss_rcnn_cls_3rd, loss_rcnn_box_3rd]
                        names = ['loss', 'loss_rpn_cls', 'loss_rpn_box', 'loss_rcnn_cls', 'loss_rcnn_box', 'loss_rcnn_cls_2nd', 'loss_rcnn_box_2nd', 'loss_rcnn_cls_3rd', 'loss_rcnn_box_3rd']
                    else:
                        scalars = [loss_temp, loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box]
                        names = ['loss', 'loss_rpn_cls', 'loss_rpn_box', 'loss_rcnn_cls', 'loss_rcnn_box']
                    write_scalars(writer, scalars, names, iters_per_epoch * (epoch - 1) + step, tag='train_loss')
                '''
                loss_temp = 0
                start = time.time()
        
        
        if args.mGPUs:
            save_name = os.path.join(output_dir, 'fpn_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        else:
            save_name = os.path.join(output_dir, 'fpn_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        _print('save model: {}'.format(save_name), )
        
        end = time.time()
        print(end - start)
    
    
    
    
    
    
    
    
    
    
    
        