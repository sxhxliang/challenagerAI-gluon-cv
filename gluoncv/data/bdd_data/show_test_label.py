import json
import argparse
from multiprocessing import Pool
import os
from os.path import exists, splitext, isdir, isfile, join, split, dirname
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties
from PIL import Image
import sys


# {  
#     "name": str,
#     "timestamp": 1000,
#     "category": str,
#     "bbox": [x1, y1, x2, y2],
#     "score": float
# }

import cv2
import random

def mask_drivable_map(img, pred_map, alpha=0.5):
    color = np.array([0,255,255])
    # colors = np.array([[0, 0, 0, 255],
    #                 [217, 83, 79, 255],
    #                 [91, 192, 222, 255]]) / 255
    # mask[:,:,0]=0
   
    mask = np.zeros((720,1280,3))
    mask[:,:,0] = pred_map==0.
    mask[:,:,1]  = pred_map==1.
    mask[:,:,2]  = pred_map==2.
    mask_color = mask * color
    
    img = np.where(mask, img * (1 - alpha) + mask_color * alpha, img)

    return img.astype('uint8')
    # drivable_maps = drivable_maps * 127.5
    #     if self._one_hot_label:
    #         drivable_maps = drivable_maps * 2
            
    #         drivable_maps = mx.nd.concat(a,b,c,dim=0)

def draw_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, ax=None,
              reverse_rgb=False, absolute_coordinates=True):
    font = cv2.FONT_HERSHEY_SIMPLEX


    if len(bboxes) < 1:
        return ax

    
    bboxes = np.array(bboxes).astype(np.float32)

    scores = np.array(scores).astype(np.float32)

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    # use random colors if None is provided
    if colors is None:
        colors = dict()
    n = 0
    for i, bbox in enumerate(bboxes):
        # print('bbox', bbox)
        # print(scores.flat[i])
        # print(labels)
        # bbox = bbox
        if scores is not None and scores.flat[i] < thresh:
            continue
        n += 1
        cls_id = labels[i]

        if cls_id not in colors:
            colors[cls_id] = (random.random()*255, random.random()*255, random.random()*255) 
        xmin, ymin, xmax, ymax = [x for x in bbox]
        # print(xmin, ymin, xmax, ymax)
        cv2.rectangle(img,(xmin, ymin), (xmax, ymax), colors[cls_id], 2)


        score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
        if cls_id or score:
            cv2.putText(img, '{:s} {:s}'.format(cls_id, score), (int(xmin), int(ymin)-2), font, 1.5, colors[cls_id], 3)
    return img


# filename = '/data1/datasets/bdd100k/test_result/det.json'
# folder_path = '/data1/datasets/bdd100k/images/100k/test2018/'
# filename = '/Volumes/DATASET/BDD100k/bdd100k/annotations/det2.json'
folder_path = '/Volumes/DATASET/BDD100k/bdd100k/images/100k/test2018/'
filename = '/Volumes/18080806004/bddresult/24result/det.json'
drivable_folder_path = '/Volumes/18080806004/bddresult/24result/seg/'
labels = json.load(open(filename,'r'))

cur_name = None
cur_img = None
cur_boxes = []
cur_scores = []
cur_cats = []

cur_is_new = False
for obj in labels:
    if cur_name == obj['name']:
        if float(obj['score']) < 0.6:
            continue
        cur_boxes.append(obj['bbox'])
        cur_scores.append(obj['score'])
        cur_cats.append(obj['category'])
    else:
        if cur_is_new:
            # print(cur_img)
            img = draw_bbox(cur_img, cur_boxes, cur_scores, cur_cats)
            # print(img)
            plt.imshow(img)
            plt.show()
        cur_is_new = True
        cur_name = obj['name']
        cur_img = Image.open(folder_path+cur_name)
        cur_img = np.array(cur_img)
        cur_mask_name = cur_name.replace('.jpg','_drivable_id.png')
        print(drivable_folder_path + cur_mask_name)
        cur_mask = Image.open(drivable_folder_path + cur_mask_name)
        cur_mask = np.array(cur_mask)
        cur_img = mask_drivable_map(cur_img, cur_mask)
        
        cur_boxes = []
        cur_scores = []
        cur_cats = []
        cur_boxes.append(obj['bbox'])
        cur_scores.append(obj['score'])
        cur_cats.append(obj['category'])


# filename = '/Volumes/DATASET/BDD100k/bdd100k/annotations/det.json'
# labels = json.load(open(filename,'r'))
# for i in range(len(labels)):
#     bbox = labels[i]['bbox']
#     labels[i]['bbox'] = (np.array(bbox) * 1.2).tolist()

# with open('/Volumes/DATASET/BDD100k/bdd100k/annotations/det.json', 'w') as jsonf:
#     json.dump(test_json, jsonf)

        
    