from pycocotools.coco import COCO

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir=os.path.join('../data/', "COCO")
dataType='train2017'
annFile=os.path.join(dataDir, "annotations", "instances_%s.json" % dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

img_ids = coco.getImgIds()
imgInfo = coco.loadImgs(img_ids[0])[0]

# cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
# print(len(nms)) # 80
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))
# print(len(nms)) # 12
#
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']); #
# print(catIds)
# imgIds = coco.getImgIds(catIds=catIds )# 包含这三个类别的图片id列表
# print(imgIds)
# img = coco.loadImgs([379520])
# print(img)
#
I = io.imread(os.path.join(dataDir, "images", "train2017", imgInfo['file_name']))#img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()

# imgIds = coco.getImgIds(imgIds = [324158])
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#
# # use url to load image
# I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.show()