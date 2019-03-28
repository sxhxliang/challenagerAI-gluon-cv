from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from PIL import Image
from PIL import ImageDraw 
import os
import argparse


parser = argparse.ArgumentParser(description='get instance images datasets ')

parser.add_argument('--dataDir', type=str, default='/Volumes/DATASET/COCODatasets/COCO2017',
                    help='directory of datasets')
parser.add_argument('--dataType', type=str, default='val2017',
                    help='datasets name')
parser.add_argument('--labelPath', type=str, default='/Volumes/DATASET/COCODatasets/COCO2017segDataset/lable',
                    help='directory of saved instance mask')
parser.add_argument('--imagePath', type=str, default='/Volumes/DATASET/COCODatasets/COCO2017segDataset/image',
                    help='directory of saved instance images')
parser.add_argument('--instanceSize', type=float, default=0.0,
                    help='directory of saved instance images')

opt = parser.parse_args()

annFile='{}/annotations/instances_{}.json'.format(opt.dataDir, opt.dataType)
dataDir = opt.dataDir
dataType = opt.dataType
image_path = opt.imagePath
label_path= opt.labelPath
instance_size = opt.instanceSize

# python genMaskAndImg.py --dataType val2017 --dataDir /data1/datasets/coco --labelPath /data1/datasets/coco/seg_val2017/label --imagePath /data1/datasets/coco/seg_val2017/image
# python genMaskAndImg.py --dataType train2017 --dataDir /data1/datasets/coco --labelPath /data1/datasets/coco/seg_train2017/label --imagePath /data1/datasets/coco/seg_train2017/image  --instanceSize 900



def saveImg(root_path, image, cat_name, img_id, ann_id):
    img_id = str(img_id)
    path = os.path.join(root_path, cat_name, str(img_id) + '_' + str(ann_id) + '.JPEG')
    if os.path.isfile(path):
        print('file have exists')
    else:
        folder_path = os.path.join(root_path, cat_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
            print('create folder', cat_name)
        # print(path)
        image.save(path)
        

def cropImg(image,ann):
    bbox = ann['bbox']
    img = Image.fromarray(image)
    return img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

def imageGenMask(imageMask,ann):
  
    imgDraw = ImageDraw.Draw(imageMask) 

    for polygon in ann['segmentation']:
        imgDraw.polygon(tuple(polygon), fill = 1)  
    # draw.polygon(box, 'olive', 'hotpink')
    # imagelabel.show()#hotpink
    bbox = ann['bbox']
#     print('id',ann['id'])
    imagecrop = imageMask.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
    imagecrop = np.array(imagecrop) * 255
    return Image.fromarray(imagecrop)



# initialize COCO api for instance annotations
coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats] 


for nm in nms:
    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=[nm]);
    # print(catIds)[1, 3, 18, 41]
    # all image ids
    imgIds = coco.getImgIds(catIds=catIds);
    # print(len(imgIds))
    # all images
    imgs = coco.loadImgs(imgIds)
    print('categorie:', nm)
    for img in imgs:
        # print(img)
        image = io.imread('%s/%s/%s'%(dataDir, dataType,img['file_name']))
        # image = io.imread(img['coco_url'])
        # print(image)
        array = np.ndarray((img['height'], img['width'], 3), np.uint8)  
        array[:, :, 0] = 0  
        array[:, :, 1] = 0  
        array[:, :, 2] = 0  
        imageMask = Image.fromarray(array)
        # get image annIds
        annIds = coco.getAnnIds(imgIds=[img['id']], catIds=catIds, iscrowd=0)
        # print('image', img['id'], annIds)
        anns = coco.loadAnns(annIds)
        # print(anns)
        for ann in anns:  
            if ann['iscrowd'] == 0 and ann['area'] >= instance_size:
                crop_img = cropImg(image, ann)
                crop_lable = imageGenMask(imageMask, ann)
                saveImg(label_path, crop_lable, nm, img['id'], ann['id'])
                saveImg(image_path, crop_img, nm, img['id'], ann['id'])
        
