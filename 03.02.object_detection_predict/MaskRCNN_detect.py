import os
import skimage.io

from keras_config import FastMaskRCNNConfig as config
from models.layers.keras import segmentation as seg

from keras_commons import visualize as visual

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')


cfg = config.FastMaskRCNNConfig_Coco_Inference()
print(cfg.NAME)

# Root directory of the project
ROOT_DIR = "../"

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "weights", "mask_rcnn_coco.h5")


# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "predictImages")

# Create model object in inference mode.
model = seg.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=cfg)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, "dog.jpg"))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visual.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            cfg.CLASS_NAMES, r['scores'])