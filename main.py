from gluoncv import model_zoo, data, utils
import mxnet as mx
import numpy as np
from PIL import Image
import json
# from tqdm import tqdm 
import os

# epoch = 1
save_path = '/data/output'
test_path = '/data/data'
test_json = []

CLASSES = ['traffic light', 'traffic sign', 'person', 'rider', 'bike', 'bus', 'car', 'motor', 'train', 'truck']
_score_thresh = 0.45
ctx = mx.gpu()
resize_map = mx.image.ForceResizeAug((1280,720), interp=2)

net = model_zoo.get_model('mask_rcnn_resnet50_v1b_bdd', pretrained=False, pretrained_base=False)
net.load_parameters('/data/bddv4_continuemask_rcnn_resnet50_v1b_bdd_0024.params')
net.collect_params().reset_ctx(ctx)

def save_drivable_map(pred_map, file_id):
    drivable_name = file_id + '_drivable_id' + '.png'

    mask = mx.nd.softmax(pred_map, axis=2)
    mask = mask>0.5
    color = np.array([0,1,2])
    mask = mask.asnumpy() * color
    mask = np.sum(mask, axis=2).astype('uint8')
    # print(mask.shape)
    img = Image.fromarray(mask)
    # img.save()
    img.save(save_path + 'seg/' + drivable_name, 'png')



if not os.path.isdir(save_path):
    os.mkdir(save_path)


seg_path = os.path.join(save_path,'seg')
if not os.path.isdir(seg_path):
    os.mkdir(seg_path)


if not os.path.isdir(test_path):
    print('images data not found in ', test_path)

else:
    for filename in os.listdir(test_path):
        file_path = os.path.join(test_path,filename)
        if not os.path.isfile(file_path):
            print(file_path,'is not a file')
            continue
        x, orig_img = data.transforms.presets.rcnn.load_test(test_path+filename, max_size=1280)

        

        ids, scores, bboxes, drivable_maps = net(x.as_in_context(ctx))
        det_id, det_score, det_bbox = [xx[0].asnumpy() for xx in [ids, scores, bboxes]]

        
        mask = drivable_maps[0].transpose((1,2,0)).as_in_context(mx.cpu())
        mask = resize_map(mask)

        save_drivable_map(mask, file_id)
        # ids, scores, bboxes

        valid = np.where(((det_id >= 0) & (det_score >= _score_thresh)))[0]
        det_id = det_id[valid]
        det_score = det_score[valid]
        det_bbox = det_bbox[valid] 
        # print(det_score.shape)
        for cid, score,  bbox in zip(det_id, det_score, det_bbox):
            # print(cid)
            test_json.append({
                    "name": filename,
                    "timestamp": 1000,
                    "category": CLASSES[int(cid[0])],
                    "bbox": bbox.tolist(),
                    "score": float(score[0])
                })
    save_json_path = os.join(save_path,'det.json')
    print('save_path',save_json_path)
    with open(save_json_path, 'w') as jsonf:
        json.dump(test_json, jsonf)

    print('finished')



