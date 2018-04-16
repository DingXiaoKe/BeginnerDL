# coding=utf-8
from os.path import join
from os import listdir
from torch.utils.data import Dataset
from torchvision.transforms.transforms import ToTensor, Normalize, Compose
from lib.utils.utils.image import is_image_file, load_img

class DataSetFromFolderForPicTransfer(Dataset):
    def __init__(self, image_dir):
        super(DataSetFromFolderForPicTransfer, self).__init__()
        self.photo_path = join(image_dir, "A")
        self.sketch_path = join(image_dir, "B")
        self.image_filenames = [x for x in listdir(self.photo_path) if is_image_file(x) ]

        transform_list = [ToTensor(),
                     Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        self.transform = Compose(transform_list)

    def __getitem__(self, index):
        input = load_img(join(self.photo_path, self.image_filenames[index]))
        input = self.transform(input)
        target = load_img(join(self.sketch_path, self.image_filenames[index]))
        target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)