import json
import os
# '/data1/datasets/bdd100k/testB_result/det.json'
# '/data1/datasets/bdd100k/testB_result/det2.json'
# '/data1/datasets/bdd100k/testB_result/det3.json'
# '/data1/datasets/bdd100k/testB_result/det4.json'

# det_final = []
# det1 = json.load(open('/data1/datasets/bdd100k/testB_result/det1.json',"r"))
# det2 = json.load(open('/data1/datasets/bdd100k/testB_result/det2.json',"r"))
# det3 = json.load(open('/data1/datasets/bdd100k/testB_result/det3.json',"r"))
# det4 = json.load(open('/data1/datasets/bdd100k/testB_result/det4.json',"r"))

# det_final.extend(det1)
# det_final.extend(det2)
# det_final.extend(det3)
# det_final.extend(det4)

dmap = os.listdir('/data1/datasets/bdd100k/testB_result/seg')

filenames = []
with open('ai_challenger_adp2018_testb_20180917.txt','r') as f:
    for file_id in f.readlines():
        file_id = file_id.replace('\n','')
        # filename = file_id+'.jpg'
        filename = file_id+ '_drivable_id' + '.png'
        filenames.append(filename)


for filename in dmap:
    # filename = item['name']
    if filename in filenames:
        print(filename)

# # print('save_path + 'det4.json'')
# with open('/data1/datasets/bdd100k/testB_result/det_final.json', 'w') as jsonf:
#     json.dump(det_final, jsonf)