# coding=utf-8
import os
import pandas as pd
import shutil
from tqdm import tqdm
DATA_PATH = os.path.join('../ganData/')
ORIGINAL_DATA_PATH = os.path.join(DATA_PATH, 'face/celebA/')
DATA_PATH = os.path.join(DATA_PATH, 'face_gender')
os.makedirs(DATA_PATH)
os.makedirs(os.path.join(DATA_PATH, "0"))
os.makedirs(os.path.join(DATA_PATH, "1"))

ATTR_FILE_PATH = os.path.join("face_attr.csv")
data = pd.read_csv(ATTR_FILE_PATH, usecols=("File_Names", "Male")).as_matrix()

pbar = tqdm(enumerate(data))

for index, record in pbar:
    pbar.set_description("loading images : %s" % (len(data) - index - 1))
    filename = record[0]
    label = record[1]
    label = 0 if label == -1 else 1
    shutil.copy(os.path.join(ORIGINAL_DATA_PATH, filename),
                os.path.join(DATA_PATH, "%s" % label))


