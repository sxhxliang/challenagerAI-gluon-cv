"""MS COCO object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import numpy as np
import mxnet as mx
from .utils import try_import_pycocotools
from ..base import VisionDataset

__all__ = ['BDDInstance']


class BDDInstance(VisionDataset):
    """MS COCO instance segmentation dataset.

    Parameters
    ----------
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of str, default ['instances_val2017']
        Json annotations name.
        Candidates can be: instances_val2017, instances_train2017.
    transform : callable, defaut None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    min_object_area : float, default is 1
        Minimum accepted ground-truth area, if an object's area is smaller than this value,
        it will be ignored.
    skip_empty : bool, default is True
        Whether skip images with no valid object. This should be `True` in training, otherwise
        it will cause undefined behavior.

    """
    CLASSES =  ['traffic light', 'traffic sign', 'person', 'rider', 'bike', 'bus', 'car', 'motor', 'train', 'truck']

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'bdd'),
                 splits=('instances_train2018',), transform=None, min_object_area=1,
                 skip_empty=True, use_color_maps=False, is_training=True):
        super(BDDInstance, self).__init__(root)
        self._is_training = is_training
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._min_object_area = min_object_area
        self._skip_empty = skip_empty
        self._use_color_maps = use_color_maps
        if isinstance(splits, mx.base.string_types):
            splits = [splits]
        self._splits = splits
        # to avoid trouble, we always use contiguous IDs except dealing with cocoapi
        self.index_map = dict(zip(type(self).CLASSES, range(self.num_class)))
        self.json_id_to_contiguous = None
        self.contiguous_id_to_json = None
        self._coco = []
        self._items, self._labels, self._drivable_maps = self._load_jsons()

    def __str__(self):
        detail = ','.join([str(s) for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def coco(self):
        """Return pycocotools object for evaluation purposes."""
        if not self._coco:
            raise ValueError("No coco objects found, dataset not initialized.")
        elif len(self._coco) > 1:
            raise NotImplementedError(
                "Currently we don't support evaluating {} JSON files".format(len(self._coco)))
        return self._coco[0]

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_path = self._items[idx]
        label = self._labels[idx]
        drivable_map_path = self._drivable_maps[idx]
        img = mx.image.imread(img_path, 1)
        drivable_map = mx.image.imread(drivable_map_path, 1)
        if self._transform is not None:
            return self._transform(img, label, segm)
        if not self._is_training:
            return img, label, drivable_map, [idx]
        return img, label, drivable_map

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []
        drivable_maps = []

        # lazy import pycocotools
        try_import_pycocotools()
        from pycocotools.coco import COCO
        for split in self._splits:
            anno = os.path.join(self._root, 'annotations', split) + '.json'
            print(anno)
            _coco = COCO(anno)
            self._coco.append(_coco)
            classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
            # print(classes)
            # print(self.classes)
            if not classes == self.classes:
                raise ValueError("Incompatible category names with COCO: ")
            assert classes == self.classes
            json_id_to_contiguous = {
                v: k for k, v in enumerate(_coco.getCatIds())}
            if self.json_id_to_contiguous is None:
                self.json_id_to_contiguous = json_id_to_contiguous
                self.contiguous_id_to_json = {
                    v: k for k, v in self.json_id_to_contiguous.items()}
            else:
                assert self.json_id_to_contiguous == json_id_to_contiguous

            # iterate through the annotations
            image_ids = sorted(_coco.getImgIds())
            for entry in _coco.loadImgs(image_ids):
                # print(entry)
                dirname, filename = entry['url'].split('/')[-2:]
                abs_path = os.path.join(self._root,'images/100k', dirname, filename)
                if not os.path.exists(abs_path):
                    raise IOError('Image: {} not exists.'.format(abs_path))
                label = self._check_load_bbox(_coco, entry)
                
                
                if self._use_color_maps:
                    label_name = 'color_labels'
                    filename = filename.replace('.jpg', '_drivable_color.png')
                else:
                    label_name = 'labels'
                    filename = filename.replace('.jpg', '_drivable_id.png')
                    
                seg_abs_path = os.path.join(self._root, 'drivable_maps',label_name, dirname, filename)
                # skip images without objects
                if self._skip_empty and label is None:
                    continue
                items.append(abs_path)
                labels.append(label)
                drivable_maps.append(seg_abs_path)

        return items, labels, drivable_maps

    def _check_load_bbox(self, coco, entry):
        """Check and load ground-truth labels"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        # valid_segs = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            if obj.get('ignore', 0) == 1:
                continue
            # crowd objs cannot be used for segmentation
            if obj.get('iscrowd', 0) == 1:
                continue
            # need accurate floating point box representation
            x1, y1, w, h = obj['bbox']
            x2, y2 = x1 + np.maximum(0, w), y1 + np.maximum(0, h)
            # clip to image boundary
            x1 = np.minimum(width, np.maximum(0, x1))
            y1 = np.minimum(height, np.maximum(0, y1))
            x2 = np.minimum(width, np.maximum(0, x2))
            y2 = np.minimum(height, np.maximum(0, y2))
            # require non-zero seg area and more than 1x1 box size
            if obj['area'] > self._min_object_area and x2 > x1 and y2 > y1 \
                    and (x2 - x1) * (y2 - y1) >= 4:
                contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
                valid_objs.append([x1, y1, x2, y2, contiguous_cid])

                # segs = obj['segmentation']
                # assert isinstance(segs, list), '{}'.format(obj.get('iscrowd', 0))
                # valid_segs.append([np.asarray(p).reshape(-1, 2).astype('float32')
                #                    for p in segs if len(p) >= 6])
        # there is no easy way to return a polygon placeholder: None is returned
        # in validation, None cannot be used for batchify -> drop label in transform
        # in training: empty images should be be skipped
        if not valid_objs:
            valid_objs = None
            # valid_segs = None
        else:
            valid_objs = np.asarray(valid_objs).astype('float32')
        return valid_objs
        # return valid_objs, valid_segs
