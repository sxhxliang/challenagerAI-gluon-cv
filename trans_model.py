import mxnet as mx
import numpy as np
model = mx.ndarray.load('/home/user/.mxnet/models/mask_rcnn_resnet50_v1b_coco-a3527fdc.params')
model_new = model.copy()
cls_names = ['class_predictor.bias', 'class_predictor.weight']
box_names = ['box_predictor.bias', 'box_predictor.weight']

inds = [0,10,12,1,1,2,6,3,4,7,8]
cls_index = mx.nd.array(inds).reshape(-1)
box_index = mx.nd.array([(i*4,i*4+1,i*4+2,i*4+3) for i in inds[1:]]).reshape(-1)
for s in cls_names:
    model_new[s] = mx.ndarray.take(model_new[s], cls_index)
    print(model_new[s].shape, model[s].shape)
    model_new[s][4] = model_new[s][4]*0.9
    
for s in box_names:
    model_new[s] = mx.ndarray.take(model_new[s], box_index)
    print(model_new[s].shape, model[s].shape)
mx.ndarray.save('/home/user/.mxnet/models/mask_rcnn_resnet50_v1b_bdd.params', model_new)