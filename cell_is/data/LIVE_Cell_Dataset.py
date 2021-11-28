# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 23:37:10 2021

@author: SuperWZL
"""

from model.config import cfg
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid
from collections import defaultdict
import time
# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

class LIVE_Cell_Dataset(COCO):
    def __init__(self, json_File_Path = None):
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not json_File_Path == None:
            print("loading json-File into memory...")
            tic = time.time()
            dataset = json.load(open(json_File_Path, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self._createIndex()
        
    def _createIndex(self):
        print('creating index...')
        anns, imgs = {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        for cat in cfg.TRAIN.CELL_TYPE:
            catToImgs[cat] = []
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[self.dataset['annotations'][ann]['image_id']].append(self.dataset['annotations'][ann])
                anns[self.dataset['annotations'][ann]['id']] = self.dataset['annotations'][ann]
                
        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img
                for cat in cfg.TRAIN.CELL_TYPE:
                    if cat in img['file_name']:
                        catToImgs[cat].append(img)
                        break
            
        print('index created!')
        
        self.anns = anns
        self.imgs = imgs
        self.catToImgs = catToImgs
        self.imgToAnns = imgToAnns
        
    def Img_Mask(self, img_id = None, show = False):
        if img_id == None:
            return 0
        anns = self.imgToAnns[img_id]
        submask_lst = []
        for ann in anns:
            tmp_mask = super().annToMask(ann)
            submask_lst.append(tmp_mask)
        ImgMask = sum(submask_lst)
        ImgMask[ImgMask>0]=1
        if show:
            plt.figure()
            plt.imshow(ImgMask)
            plt.show()
        return ImgMask
    
    def getImgPath(self, img_ids = []):
        if len(img_ids) == 0:
            return 0
        Path = []
        for img_id in img_ids:
            img = self.imgs[img_id]
            img_path = os.path.join(cfg.PATH.LIVE_TRAIN_VAL_IMAGES, img['file_name'])
            Path.append(img_path)
        return Path
            
    def getbbox(self, img_id = None):
        if img_id == None:
            return 0
        Bbox_lst = []
        anns = self.imgToAnns(img_id)
        for ann in anns:
            bbox = ann['bbox']
            Bbox_lst.append(bbox)
        return Bbox
            
        
if __name__ == "__main__":
    live_celldata = LIVE_Cell_Dataset(cfg.PATH.LIVE_TRAIN_JSON)
    
    

