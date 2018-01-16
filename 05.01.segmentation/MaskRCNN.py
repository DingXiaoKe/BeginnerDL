from keras_config import FastMaskRCNNConfig as config
import os
from keras_datareaders.FastMaskRCNN import CocoDataset
import numpy as np
from keras_models import segmentation as seg
from keras_commons import fastmaskrcnn as utils
import time
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

cfg = config.FastMaskRCNNConfig_Coco()
ROOT_DIR = "../"
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "weights", "mask_rcnn_coco.h5")

model = seg.MaskRCNN(mode="training", config=cfg,
                          model_dir="../logs")
model.load_weights(COCO_MODEL_PATH, by_name=True)
dataset_train = CocoDataset()
dataset_train.load_coco("../data/COCO", "train", year=2017, auto_download=False)
dataset_train.load_coco("../data/COCO", "val", year=2017, auto_download=False)
dataset_train.prepare()

# Validation dataset
dataset_val = CocoDataset()
dataset_val.load_coco("../data/COCO", "val", year=2017, auto_download=False)
dataset_val.prepare()

# *** This training schedule is an example. Update to your needs ***

# Training - Stage 1
print("Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=cfg.LEARNING_RATE,
            epochs=40,
            layers='heads')

# Training - Stage 2
# Finetune layers from ResNet stage 4 and up
print("Fine tune Resnet stage 4 and up")
model.train(dataset_train, dataset_val,
            learning_rate=cfg.LEARNING_RATE,
            epochs=120,
            layers='4+')

# Training - Stage 3
# Fine tune all layers
print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=cfg.LEARNING_RATE / 10,
            epochs=160,
            layers='all')

def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


cfg = config.FastMaskRCNNConfig_Coco_Inference()
model = seg.MaskRCNN(mode="inference", config=cfg,
                          model_dir="../logs")
model.load_weights(COCO_MODEL_PATH, by_name=True)
dataset_val = CocoDataset()
coco = dataset_val.load_coco("../data/COCO", "val", year=2017, return_coco=True,
                             auto_download=False)
dataset_val.prepare()
evaluate_coco(model, dataset_val, coco, "bbox", limit=int(500))



image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
inference_config = cfg
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        utils.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(utils.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id,
                         r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)

print("mAP: ", np.mean(APs))
