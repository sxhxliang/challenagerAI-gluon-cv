from __future__ import division
import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import argparse
import glob
import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import nd
from mxnet import gluon
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv.data import batchify
from gluoncv.utils.metrics.bdd_detection import BDDDetectionMetric
from gluoncv.utils.metrics.bdd_instance import BDDInstanceMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.data.transforms.presets.rcnn import BDDMaskRCNNDefaultValTransform
from mxnet.gluon.data.vision.transforms import Resize
import json
from PIL import Image
def parse_args():
    parser = argparse.ArgumentParser(description='Validate Mask RCNN networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Base feature extraction network name")
    parser.add_argument('--dataset', type=str, default='bdd',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-json', action='store_true',
                        help='Save coco output json')
    parser.add_argument('--eval-all', action='store_true',
                        help='Eval all models begins with save prefix. Use with pretrained.')
    args = parser.parse_args()
    return args
save_path = '/data1/datasets/bdd100k/val_result2/'
CLASSES = ['traffic light', 'traffic sign', 'person', 'rider', 'bike', 'bus', 'car', 'motor', 'train', 'truck']

def get_dataset(dataset, args):
    if dataset.lower() == 'bdd':
        val_dataset = gdata.BDDInstance(root='/data1/datasets/bdd100k/', splits='bdd100k_to_coco_labels_images_val2018', skip_empty=False, use_color_maps=False, is_training=False)
        val_metric = BDDInstanceMetric(val_dataset, args.save_prefix + '_eval', cleanup=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric

def get_dataloader(net, val_dataset, batch_size, num_workers):
    """Get dataloader."""
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(BDDMaskRCNNDefaultValTransform(net.short, net.max_size)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)
    return val_loader

def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


def save_drivable_map(pred_map, file_id):
    drivable_name = file_id + '_drivable_id' + '.png'
    mask = pred_map>0.5
    color = np.array([0,1,2])
    mask = mask * color
    mask = np.sum(mask, axis=2).astype('uint8')
    # print(mask.shape)
    img = Image.fromarray(mask)
    # img.save()
    img.save(save_path + 'seg/' + drivable_name, 'png')

def validate(net, val_data, ctx, eval_metric, size):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    resize_map = mx.image.ForceResizeAug((1280,720), interp=2)
    eval_metric.reset()
    net.hybridize(static_alloc=True)
    val_json = []
   
    with tqdm(total=size) as pbar:
        for ib, batch in enumerate(val_data, start=9500):
            # batch = split_and_load(batch, ctx_list=ctx)
            # det_bboxes = []
            # det_ids = []
            # det_scores = []
            # gt_bboxes = []
            # gt_ids = []
            # gt_difficults = []
            # for x, y, im_scale in zip(*batch):
            #     # get prediction results
            #     ids, scores, bboxes, masks = net(x)
            #     det_ids.append(ids)
            #     det_scores.append(scores)
            #     # clip to image size
            #     det_bboxes.append(clipper(bboxes, x))
            #     # rescale to original resolution
            #     im_scale = im_scale.reshape((-1)).asscalar()
            #     det_bboxes[-1] *= im_scale
            #     # split ground truths
            #     gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            #     gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            #     gt_bboxes[-1] *= im_scale
            #     gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
            # # update metric
            # for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults):
            #     eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
            # pbar.update(len(ctx))

            batch = split_and_load(batch, ctx_list=ctx)
            det_bboxes = []
            det_ids = []
            det_scores = []
            drivable_masks = []
            det_infos = []
            file_ids = []
            for x, im_info, file_id in zip(*batch):
                # get prediction results
                # print(file_id)
                ids, scores, bboxes, masks = net(x)
                det_bboxes.append(clipper(bboxes, x))
                det_ids.append(ids)
                det_scores.append(scores)
                drivable_masks.append(masks)
                det_infos.append(im_info)
                file_ids.append(file_id)
                
            # update metric
            for det_bbox, det_id, det_score, drivable_mask, det_info, file_id in zip(det_bboxes, det_ids, det_scores, drivable_masks, det_infos, file_ids):
                # print('det_bbox', det_bbox.shape) # (1, 1000, 4)
                # print('det_info', det_info) # (1,3) [[7.20e+02 1.28e+03 1.00e+00]]
                
                for i in range(det_info.shape[0]):
                    # numpy everything
                    det_bbox = det_bbox[i].asnumpy() #(1000, 4)
                    det_id = det_id[i].asnumpy() #(1000, 1)
                    det_score = det_score[i].asnumpy() #(1000, 1)
                    drivable_mask = drivable_mask[i] #(3, 180, 320)
                    det_info = det_info[i].asnumpy() #(3)

                    drivable_mask = mx.nd.softmax(drivable_mask, axis=0).transpose((1,2,0)).as_in_context(mx.cpu())
                    drivable_mask = resize_map(drivable_mask).asnumpy()

                    file_id = str(file_id[i].asnumpy()[0])
                    # 保存 图片
                    save_drivable_map(drivable_mask, file_id)


                    # filter by conf threshold
                    im_height, im_width, im_scale = det_info
                    valid = np.where(((det_id >= 0) & (det_score >= 0.001)))[0]
                    det_id = det_id[valid]
                    det_score = det_score[valid]
                    det_bbox = det_bbox[valid] / im_scale

                    for cid, score,  bbox in zip(det_id, det_score, det_bbox):
                         # print(cid)
                        val_json.append({
                            "name": file_id,
                            "timestamp": 1000,
                            "category": CLASSES[int(cid[0])],
                            "bbox": bbox.tolist(),
                            "score": float(score[0])
                        })
                    # det_mask = det_mask[valid]
                    # fill full mask
                    # im_height, im_width = int(round(im_height / im_scale)), int(round(im_width / im_scale))
                    # full_masks = []
                    # for bbox, mask in zip(det_bbox, det_mask):
                    #     full_masks.append(gcv.data.transforms.mask.fill(mask, bbox, (im_width, im_height)))
                    # full_masks = np.array(full_masks)
                    # print(det_bbox)

                    eval_metric.update(det_bbox, det_id, det_score, drivable_mask)

            pbar.update(len(ctx))
    print(save_path + 'val.json')
    with open(save_path +'val.json', 'w') as jsonf:
        json.dump(val_json, jsonf)
    return eval_metric.get()

if __name__ == '__main__':
    args = parse_args()

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx)  # 1 batch per device

    # network
    net_name = '_'.join(('mask_rcnn', args.network, args.dataset))
    args.save_prefix += net_name
    # if args.pretrained.lower() in ['true', '1', 'yes', 't']:
    #     net = gcv.model_zoo.get_model(net_name, pretrained=True)
    # else:
    net = gcv.model_zoo.get_model(net_name, pretrained=False, pretrained_base=False)
    net.load_parameters('deformablemask_rcnn_resnet50_v1b_bdd_0019.params')
    net.collect_params().reset_ctx(ctx)

    # training data
    val_dataset, eval_metric = get_dataset(args.dataset, args)
    val_data = get_dataloader(
        net, val_dataset, args.batch_size, args.num_workers)

    # validation
    if not args.eval_all:
        names, values = validate(net, val_data, ctx, eval_metric, len(val_dataset))
        for k, v in zip(names, values):
            print(k, v)
    else:
        saved_models = glob.glob(args.save_prefix + '*.params')
        for epoch, saved_model in enumerate(sorted(saved_models)):
            print('[Epoch {}] Validating from {}'.format(epoch, saved_model))
            net.load_parameters(saved_model)
            net.collect_params().reset_ctx(ctx)
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric, len(val_dataset))
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            print('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
            with open(args.save_prefix+'_best_map.log', 'a') as f:
                f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
