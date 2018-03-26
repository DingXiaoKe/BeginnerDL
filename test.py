import numpy as np
import os
import shutil

SOURCE_PATH = [os.path.join("data/STL/train/1"), os.path.join("data/STL/test/1")]
TARGET_PATH = os.path.join("ganData/STL/airplane")

index = 0
for path in SOURCE_PATH:
    for file in os.listdir(path):
        os.rename(os.path.join(path, file), os.path.join(TARGET_PATH, "%d.jpg" % index))
        # shutil.move(os.path.join(path, "%d.jpg" % index), TARGET_PATH)
        index = index + 1
