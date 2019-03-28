from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
import mxnet as mx
import numpy as np
from PIL import Image
import json
from tqdm import tqdm 

# epoch = 1
save_path = '/data1/datasets/bdd100k/testB_result/'
test_path = '/data1/datasets/bdd100k/images/100k/test2018/'
test_json = []

CLASSES = ['traffic light', 'traffic sign', 'person', 'rider', 'bike', 'bus', 'car', 'motor', 'train', 'truck']
_score_thresh = 0.5
ctx = mx.gpu(3)
resize_map = mx.image.ForceResizeAug((1280,720), interp=2)

net = model_zoo.get_model('mask_rcnn_resnet50_v1b_bdd', pretrained=False, pretrained_base=False)
net.load_parameters('bddv4_continuemask_rcnn_resnet50_v1b_bdd_0024.params')
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


#  ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
with open('ai_challenger_adp2018_testb_20180917_t4.txt','r') as f:
    for file_id in tqdm(f.readlines()):
        file_id = file_id.replace('\n','')
        filename = file_id+'.jpg'
        x, orig_img = data.transforms.presets.rcnn.load_test(test_path+filename, max_size=1280)
   
        

        ids, scores, bboxes, drivable_maps = net(x.as_in_context(ctx))
        det_id, det_score, det_bbox = [xx[0].asnumpy() for xx in [ids, scores, bboxes]]

        
        mask = drivable_maps[0].transpose((1,2,0)).as_in_context(mx.cpu())
        mask = resize_map(mask)
        # 保存 图片
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
    print(save_path + 'det4.json')
    with open( save_path +'det4.json', 'w') as jsonf:
        json.dump(test_json, jsonf)



