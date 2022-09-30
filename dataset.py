import os
import sys

import numpy as np
# import cv2
from PIL import Image
import pandas as pd

from torch.utils.data import Dataset, DataLoader, Subset
import torch
from torchvision import transforms

from utils import build_transform_from_cfg




def get_dataloader(dataset, data_dir, ann_path, mode, pipeline, csv, batch_size, num_workers, small_set = None):
    if mode == 'train':
        is_training = True
        shuffle = True
    elif mode == 'test':
        is_training = False
        shuffle = False
    elif mode == 'valid':
        is_training = False
        shuffle = False
    else:
        raise ValueError

    df = pd.read_parquet(ann_path, engine='pyarrow')

    transform = build_transform_from_cfg(pipeline)

    #dataset = CoronaryArteryDataset(df, data_dir, transform, is_training = is_training)
    dataset = getattr(sys.modules[__name__], dataset)(df, data_dir, transform, is_training = is_training, csv=csv)

    if small_set:
        dataset = Subset(dataset, indices=np.linspace(start=0, stop=len(dataset), num = 8, endpoint= False, dtype=np.uint8))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle = shuffle,
        num_workers=num_workers,
        drop_last = False
    )

    return dataloader

       

class FeatureSwapDataset(Dataset):
    def __init__(self, df, data_dir, transform, is_training=False, csv=False):
        super(FeatureSwapDataset, self).__init__()
        self.data = df.values
        self.transform = transform
        self.is_training = is_training

        self.data_dir = data_dir
        self.csv = csv

        self.range_min = 20
        self.range_max = 100
        self.range_max -= self.range_min


    def __len__(self):
        return len(self.data) 


    def __getitem__(self, idx):

        # notation (age: y1, sex:y2 (0: female, 1:male))
        dcm_path, y1, y2, y3, dcm_path2, y4, y5, y6 = self.data[idx] # score dynamic range (600-3000)

        # age normalization
        actual_y1 = y1
        y1 = (y1 - self.range_min) / self.range_max

        array_path = os.path.join(self.data_dir,dcm_path.replace('.dcm','.npy'))

        image = np.load(array_path)

        image = image - image.min()
        image = image / image.max()


        transformed = self.transform({'image':image})
        image = transformed['image']
        x = torch.cat([image,image, image], dim = 0).to(torch.float32)

        y1 = torch.FloatTensor([y1])
        y2 = torch.LongTensor([y2]).squeeze()

        ret = {
            'image' : x,
            'gt_age' :y1,
            'gt_age_int' : actual_y1,
            'gt_sex' : y2,
            'f_name' : array_path
        }

        return ret
    
if __name__ == '__main__':
    
    pass