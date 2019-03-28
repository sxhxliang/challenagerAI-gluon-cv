from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils


net = model_zoo.get_model('mask_rcnn_resnet50_v1b_bdd', pretrained=False)
net.load_parameters('bddv3onehotmask_rcnn_resnet50_v1b_bdd_0004.params')

x, orig_img = data.transforms.presets.rcnn.load_test('0a0eaeaf-9ad0c6dd.jpg',max_size=1280)

def plot_drivable_map(img, pred_map, alpha=0.5):
    mask = mx.nd.softmax(pred_map, axis=2)
    mask = mask>0.5
    color = np.array([0,255,255])
    mask[:,:,0]=0
    img = np.where(mask.asnumpy(), img * (1 - alpha) + color * alpha, img)
    return img.astype('uint8')
    
re_size = mx.image.ForceResizeAug((1280,720), interp=2)

ids, scores, bboxes, drivable_maps = net(x)
ids, scores, bboxes = [xx[0].asnumpy() for xx in [ids, scores, bboxes]]
print('obj:',len(scores))
print('drivable_maps:', drivable_maps.shape)

CLASSES = ['traffic light', 'traffic sign', 'person', 'rider', 'bike', 'bus', 'car', 'motor', 'train', 'truck']

drivable_maps = drivable_maps[0]
mask = drivable_maps.transpose((1,2,0)).as_in_context(mx.cpu())
mask = re_size(mask)
orig_img = plot_drivable_map(orig_img, mask) 

# identical to Faster RCNN object detection
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                         class_names=CLASSES, ax=ax)
plt.show()
plt.save('tessstttyyy.png')

