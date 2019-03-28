
# coding: utf-8

# In[7]:


# 1. Predict with pre-trained Mask RCNN models
# ===============================================
# 
# This article shows how to play with pre-trained Mask RCNN model.
# 
# Mask RCNN networks are extensions to Faster RCNN networks.
# :py:class:`gluoncv.model_zoo.MaskRCNN` is inherited from
# :py:class:`gluoncv.model_zoo.FasterRCNN`.
# It is highly recommended to read :doc:`../examples_detection/demo_faster_rcnn` first.
# 
# First let's import some necessary libraries:
# 
# 

# In[28]:


import mxnet as mx
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from moviepy.editor import VideoFileClip, VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
import cv2


# Load a pretrained model
# -------------------------
# 
# Let's get an Mask RCNN model trained on COCO dataset with ResNet-50 backbone.
# By specifying ``pretrained=True``, it will automatically download the model
# from the model zoo if necessary. For more pretrained models, please refer to
# :doc:`../../model_zoo/index`.
# 
# The returned model is a HybridBlock :py:class:`gluoncv.model_zoo.MaskRCNN`
# with a default context of `cpu(0)`.
# 
# 

# In[83]:


ctx = mx.gpu(3)
name = 'bddv3onehotmask_rcnn_resnet50_v1b_bdd_0010'
net = model_zoo.get_model('mask_rcnn_resnet50_v1b_bdd', pretrained=False,pretrained_base=False, ctx=ctx)
net.load_parameters(name + '.params',ctx=ctx)


# Pre-process an image
# --------------------
# 
# The pre-processing step is identical to Faster RCNN.
# 
# Next we download an image, and pre-process with preset data transforms.
# The default behavior is to resize the short edge of the image to 600px.
# But you can feed an arbitrarily sized image.
# 
# You can provide a list of image file names, such as ``[im_fname1, im_fname2,
# ...]`` to :py:func:`gluoncv.data.transforms.presets.rcnn.load_test` if you
# want to load multiple image together.
# 
# This function returns two results. The first is a NDArray with shape
# `(batch_size, RGB_channels, height, width)`. It can be fed into the
# model directly. The second one contains the images in numpy format to
# easy to be plotted. Since we only loaded a single image, the first dimension
# of `x` is 1.
# 
# Please beware that `orig_img` is resized to short edge 600px.
# 
# 

# In[84]:


# myclip = VideoFileClip('file3763.mov')
# iter_frames = myclip.iter_frames()


# In[85]:


from gluoncv.data.transforms import image as timage
def load_test_from_numpy(np_array, short=720, max_size=1280, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225),ctx=mx.cpu()):
    
    img = mx.nd.array(np_array)
    tensors = []
    origs = []
    
    img = timage.resize_short_within(img, short, max_size)
    orig_img = img.asnumpy().astype('uint8')
    img = mx.nd.image.to_tensor(img,ctx=ctx)
    img = mx.nd.image.normalize(img, mean=mean, std=std)
    tensors.append(img.expand_dims(0))
    origs.append(orig_img)

    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


# In[95]:


# img_array = next(iter_frame)
# x, orig_img = load_test_from_numpy(img_array,max_size=1280,ctx=ctx)


# # In[100]:


# print(x.context)
# ids, scores, bboxes, drivable_maps = net(x.as_in_context(ctx))

# print(ids.context, scores.context, bboxes.context, drivable_maps.context)


# # In[105]:


# ids[0].asnumpy()


# In[106]:


# ids, scores, bboxes = [xx[0].asnumpy() for xx in [ids, scores, bboxes]]

# ids = ids[0].asnumpy()
# scores = scores[0].asnumpy()
# bboxes = bboxes[0].asnumpy()

# print('obj:',len(scores))
# print('drivable_maps:', drivable_maps.shape)
# CLASSES = ['traffic light', 'traffic sign', 'person', 'rider', 'bike', 'bus', 'car', 'motor', 'train', 'truck']


# In[110]:


# drivable_maps = drivable_maps[0]
# # drivable_maps = drivable_maps.transpose((1,2,0))
# drivable_maps.shape
# ids = ids[0].asnumpy()
# scores = scores[0].asnumpy()
# bboxes = bboxes[0].asnumpy()# .asnumpy()


# In[89]:


# drivable_maps.shape
# plt.imshow(drivable_maps)
# plt.show()


# In[90]:


import cv2
import random
def draw_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, ax=None,
              reverse_rgb=False, absolute_coordinates=True):
    """Visualize bounding boxes.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituded.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).

    Returns
    -------
    matplotlib axes
        The ploted axes.

    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))


    if len(bboxes) < 1:
        return ax

    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()

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

        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        n += 1
        cls_id = int(labels.flat[i]) if labels is not None else -1

        if cls_id not in colors:
            colors[cls_id] = (random.random()*255, random.random()*255, random.random()*255) 
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        cv2.rectangle(img,(xmin, ymin), (xmax, ymax), colors[cls_id], 2)

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
        if class_name or score:
            cv2.putText(img, '{:s} {:s}'.format(class_name, score), (xmin, ymin-2), font, 1.5, colors[cls_id], 3)
    return img


# In[91]:


def plot_drivable_map(img, pred_map, alpha=0.5):
    mask = mx.nd.softmax(pred_map, axis=2)
    mask = mask>0.5
    color = np.array([0,255,255])
    mask[:,:,0]=0
    img = np.where(mask.asnumpy(), img * (1 - alpha) + color * alpha, img)
    return img.astype('uint8')


# In[92]:


# orig_img_2 = orig_img.copy()
# # bbox = [744.73303 ,200.20154, 956.48267, 337.25403]
# # xmin, ymin, xmax, ymax = [int(x) for x in bbox]
# # cv2.rectangle(orig_img_2,(xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
# # font = cv2.FONT_HERSHEY_SIMPLEX
# # # cv2.floodFill(orig_img_2,)
# # cv2.putText(orig_img_2, '{:s} {:s}'.format('dog', '0.2'), (xmin, ymin-2), font, 1.5,(0, 0, 255) , 3)
# # cv2.rectangle(im, (10, 10), (110, 110), (0, 0, 255), thickness=3)
# CLASSES = ['traffic light', 'traffic sign', 'person', 'rider', 'bike', 'bus', 'car', 'motor', 'train', 'truck']
# #                 9               11            0        0        1       5      2       3        6       7
# #                10               12            1        1        2       6      3       4        7       8
# CLASSES_COLOR = [(248,169,52), (220,218,46), (218,26,64), (240,26,10), (127,66,127), (6,61,99), (3,15,139), (33,130,250), (222,163,50), (1,3,69)]
# CLASSES_IDS = [i for i in range(1,11)]
# colors = dict(zip(CLASSES_IDS, CLASSES_COLOR))
# res_img = draw_bbox(orig_img_2, bboxes, scores, ids, thresh=0.5,class_names=CLASSES,colors=colors)
# orig_img_2.shape
# plt.imshow(orig_img_2)
# plt.show()


# Inference and display
# ---------------------
# 
# The Mask RCNN model returns predicted class IDs, confidence scores,
# bounding boxes coordinates and segmentation masks.
# Their shape are (batch_size, num_bboxes, 1), (batch_size, num_bboxes, 1)
# (batch_size, num_bboxes, 4), and (batch_size, num_bboxes, mask_size, mask_size)
# respectively. For the model used in this tutorial, mask_size is 14.
# 
# Object Detection results
# 
# We can use :py:func:`gluoncv.utils.viz.plot_bbox` to visualize the
# results. We slice the results for the first image and feed them into `plot_bbox`:
# 
# Plot Segmentation
# 
# :py:func:`gluoncv.utils.viz.expand_mask` will resize the segmentation mask
# and fill the bounding box size in the original image.
# :py:func:`gluoncv.utils.viz.plot_mask` will modify an image to
# overlay segmentation masks.
# 
# 

# In[ ]:


duration = 60*3

CLASSES = ['traffic light', 'traffic sign', 'person', 'rider', 'bike', 'bus', 'car', 'motor', 'train', 'truck']
#                 9               11            0        0        1       5      2       3        6       7
#                10               12            1        1        2       6      3       4        7       8
CLASSES_COLOR = [(248,169,52), (220,218,46), (218,26,64), (240,26,10), (127,66,127), (6,61,99), (3,15,139), (33,130,250), (222,163,50), (1,3,69)]
CLASSES_IDS = [i for i in range(1,11)]
colors = dict(zip(CLASSES_IDS, CLASSES_COLOR))

re_size = mx.image.ForceResizeAug((1280,720), interp=2)

myclip = VideoFileClip('file3763.mov')
iter_frames = myclip.iter_frames()
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
def make_frame(t):
    orig_img = next(iter_frames)

    x, orig_img = load_test_from_numpy(orig_img, ctx=ctx)
    
    ids, scores, bboxes, drivable_maps = net(x.as_in_context(ctx))
    
#     ids, scores, bboxes = [xx[0].asnumpy() for xx in [ids, scores, bboxes]]
    ids = ids[0].asnumpy()
    scores = scores[0].asnumpy()
    bboxes = bboxes[0].asnumpy()
    drivable_maps = drivable_maps[0]
#     print(ids.context, scores.context, bboxes.context, drivable_maps.context)
    mask = drivable_maps.transpose((1,2,0)).as_in_context(mx.cpu())
    mask = re_size(mask)
    
    orig_img = plot_drivable_map(orig_img, mask) 
    res_img = draw_bbox(orig_img, bboxes, scores, ids, thresh=0.5, class_names=CLASSES,colors=colors)
    
    
    return res_img

animation = VideoClip(make_frame, duration=duration)
animation.write_videofile(name + '.mp4', fps=30)

