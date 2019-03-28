"""1. Predict with pre-trained Mask RCNN models
===============================================

This article shows how to play with pre-trained Mask RCNN model.

Mask RCNN networks are extensions to Faster RCNN networks.
:py:class:`gluoncv.model_zoo.MaskRCNN` is inherited from
:py:class:`gluoncv.model_zoo.FasterRCNN`.
It is highly recommended to read :doc:`../examples_detection/demo_faster_rcnn` first.

First let's import some necessary libraries:
"""

from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
import mxnet as mx
######################################################################
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

# net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True, root='/Volumes/DATASET/mxnet')
net = model_zoo.get_model('mask_rcnn_resnet50_v1b_bdd', pretrained=False)
ctx = mx.gpu(0)
for param in net.collect_params().values():
    if param._data is not None:
        continue
    param.initialize(ctx)
# net.load('/Volumes/DATASET/mxnet/mask_rcnn_resnet50_v1b_bdd.')
net.load_parameters('bddv4_continuemask_rcnn_resnet50_v1b_bdd_0024.params', allow_missing=True, ignore_extra=True )

net.collect_params().reset_ctx(ctx)
######################################################################
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

# im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
#                           'gluoncv/detection/biking.jpg?raw=true',
#                           path='0a0eaeaf-9ad0c6dd.jpg')
x, orig_img = data.transforms.presets.rcnn.load_test('0a0eaeaf-9ad0c6dd.jpg',max_size=1280)

######################################################################
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
ids, scores, bboxes, masks = net(x)
ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in [ids, scores, bboxes]]
print('obj:',len(scores))
CLASSES = ['traffic light', 'traffic sign', 'person', 'rider', 'bike', 'bus', 'car', 'motor', 'train', 'truck']
#                 9               11            0        0        1       5      2       3        6       7
#                10               12            1        1        2       6      3       4        7       8
# ['person',         'bicycle',       'car',   'rider', 'bike', 'bus', 'car', 'motor', 'train', 'truck']

# CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
#                'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
#                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#                'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
#                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
#                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
#                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
#                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# paint segmentation mask on images directly
width, height = orig_img.shape[1], orig_img.shape[0]
# masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
# orig_img = utils.viz.plot_mask(orig_img, masks)

# identical to Faster RCNN object detection
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                         class_names=CLASSES, ax=ax)
plt.show()
