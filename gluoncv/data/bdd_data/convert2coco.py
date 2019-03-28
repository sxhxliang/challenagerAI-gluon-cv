from label import labels
import json
import os
from os import path as osp


class BDD_100K():
    def __init__(self, datapath):
        self.info = {"year" : 2018,
                     "version" : "1.0",
                     "description" : "BDD_100K",
                     "contributor" : "somebody",
                     "url" : "http://bdd-data.berkeley.edu/",
                     "date_created" : "2018"
                    }
        self.licenses = [{"id": 1,
                          "name": "Attribution-NonCommercial",
                          "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                         }]
        self.type = "instances"
        self.datapath = datapath
        # self.seqs = yaml.load(open(os.path.join(self.datapath, "Annotations", "db_info.yml"),
        #                            "r")
        #                      )["sequences"]

        # self.categories = [{"id": seqId+1, "name": seq["name"], "supercategory": seq["name"]}
        #                       for seqId, seq in enumerate(self.seqs)]
        # self.cat2id = {cat["name"]: catId+1 for catId, cat in enumerate(self.categories)}
        self.categories = [{"id": l.trainId, "name": l.name, "supercategory": l.category} for l in labels]
        self.cat2id = dict([(l.name, l.trainId) for l in labels])
        self.image_id = 0
        self.ann_id = 0
        self.images_folder = None
        for s,f in zip(["bdd100k_labels_images_train"],['train2018']):  
            self.images_folder = f
            #  "bdd100k_labels_images_val"
            #imlist = np.genfromtxt( os.path.join(self.datapath, "ImageSets", imageres, s + ".txt"), dtype=str)
            images, annotations = self.__get_bdd_annotation__(s)
            json_data = {"info" : self.info,
                         "images" : images,
                         "licenses" : self.licenses,
                         "type" : self.type,
                         "annotations" : annotations,
                         "categories" : self.categories}

            with open(os.path.join(self.datapath, "origin_" +
                                   s+".json"), "w") as jsonfile:
                json.dump(json_data, jsonfile, sort_keys=True, indent=4)


    def __get_image_annotation_pairs__(self, label):
        image_info, image_annotations = [], []

        self.image_id +=1
        # print('name', label['name'])
        image_info.append({"date_captured" : "2018",
                        "file_name" : label['name'],
                        "id" : self.image_id,
                        "license" : 1,
                        "url" : self.images_folder + "/" + label['name'],
                        "height" : 720,
                        "width" : 1280})
        # {
        #     "category": "traffic sign",
        #     "attributes": {
        #         "occluded": false,
        #         "truncated": false,
        #         "trafficLightColor": "none"
        #     },
        #     "manualShape": true,
        #     "manualAttributes": true,
        #     "box2d": {
        #         "x1": 1000.698742,
        #         "y1": 281.992415,
        #         "x2": 1040.626872,
        #         "y2": 326.91156
        #     },
        #     "id": 0
        # },  

        # [x1, y1] is the top left corner of the bounding box and
        # [x2, y2] the lower right.
        for obj in label['labels']:
            if 'box2d' not in obj:
                # print(label['name'])
                continue
            xy = obj['box2d']
            if xy['x1'] >= xy['x2'] and xy['y1'] >= xy['y2']:
                continue
            if obj['category'] not in self.cat2id:
                continue
            x1, x2 = min(xy['x1'], xy['x2']), max(xy['x1'], xy['x2'])
            y1, y2 = min(xy['y1'], xy['y2']), max(xy['y1'], xy['y2'])
            w, h = (x2-x1)+1, (y2-y1)+1
            self.ann_id += 1
            image_annotations.append({"segmentation" : [[x1,y1,x1,y2,x2,y2,x2,y1]],#[[x1,y1,x1,x2,y2,y2,x2,y1]],
                                "area" : w*h,
                                "iscrowd" : 0,
                                "image_id" : self.image_id,
                                "bbox" : [x1, y1, w, h],
                                'score': 1.0,
                                "category_id" : self.cat2id[obj['category']],
                                "id": self.ann_id})
        return image_info, image_annotations


    def __get_bdd_annotation__(self, name):
        # if not osp.exists(label_dir):
        #     print('Can not find', label_dir)
        #     return
        # print('Processing', label_dir)
        # label_dir = self.datapath + name
        images, annotations = [], []
        input_names = self.datapath + name + '.json'
        images_annotations = json.load(open(input_names, 'r'))
        print(len(images_annotations))
        for i in range(len(images_annotations)):
            # ann_dict = images_annotations[i]
            # if self.image_id > 9998:
                # print(self.image_id , ann_dict ,'---------------------')
            images_out, annotations_out = self.__get_image_annotation_pairs__(images_annotations[i])
            images.extend(images_out)
            annotations.extend(annotations_out)
            if self.image_id % 1000 == 0:
                print('Finished ', self.image_id, 'i', i)

        return images, annotations

if __name__ == "__main__":
    datapath ="/data1/datasets/bdd100k/bdd100k_labels_release/bdd100k/labels/"
    BDD_100K(datapath)

