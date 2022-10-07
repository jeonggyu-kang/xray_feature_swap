import os
import sys
import pickle

import numpy as np
# import cv2
from PIL import Image
import pandas as pd

from torch.utils.data import Dataset, DataLoader, Subset
import torch
from torchvision import transforms

from utils import build_transform_from_cfg




def get_dataloader(dataset, data_dir, ann_path, mode, pipeline, csv, pickle_path,batch_size, num_workers, small_set = None):
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
    dataset = getattr(sys.modules[__name__], dataset)(df, data_dir, transform, is_training = is_training, csv=csv, pickle_path = pickle_path)

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


class AGEDataset(Dataset):
    def __init__(self, df, data_dir, transform, is_training=False, csv=False):
        super(AGEDataset, self).__init__()
        self.data = df.values
        self.transform = transform
        self.is_training = is_training

        self.data_dir = data_dir
        self.csv = csv

        # TODO: modify
 
        self.range_min = 20
        self.range_max = 80
        self.range_max -= self.range_min


    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        dcm_path,  score = self.data[idx] # score dynamic range (600-3000)
        #score = int(score.replace('Y', ''))  #! 1080ti server code

        score = (score - self.range_min) / self.range_max


        array_path = os.path.join(self.data_dir,dcm_path.replace('.dcm','.npy'))

        image = np.load(array_path)

        image = image - image.min()
        image = image / image.max()



        transformed = self.transform({'image':image})
        image = transformed['image']
        x = torch.cat([image,image, image], dim = 0).to(torch.float32)


        y = torch.FloatTensor([score])

        if self.csv:
            ret = {}
            ret['x'] = x
            ret['y'] = y
            ret['f_name'] = array_path
            return ret
        else:
            return x, y
       

class VariableAgeDataset(AGEDataset):
    def __init__(self, df, data_dir, transform, is_training=False, csv=False, **kwargs):
        super(VariableAgeDataset, self).__init__(df, data_dir, transform, is_training, csv)

        self._load_csv_title(**kwargs)

    def _load_csv_title(self, **kwargs):
        pickle_path = kwargs.get('pickle_path')
        with open(pickle_path, 'rb') as f:
            self.csv_title = pickle.load(f)

        self.offset = len(self.csv_title) // 2
        print(self.csv_title)

    def _make_sample_dict(self, sample):
        ret = {}
        for i in range(len(self.csv_title)):
            if i == 0 or i == self.offset: # file_name
                continue

            ret[self.csv_title[i]] = sample[i]

        return ret

    def _convert_to_tensor(self, data_dict):
        ret = {}

        for k, v in data_dict.items():
            if 'age' == k[:3].lower():
                _v = float(v)
                v =(_v - self.range_min) / self.range_max
                ret[k] = torch.FloatTensor([v])  # 0 ~ 1.0
                ret['actual_'+k] = int(_v)

            elif 'sex' == k[:3].lower():
                ret[k]  = torch.LongTensor([int(v)]).squeeze()

            elif 'cac' == k[:3].lower():
                ret[k] =  torch.LongTensor([int(v)]).squeeze()

            else:
                raise NotImplementedError('{} is not supported!'.format(k))

        return ret        


    def __getitem__(self, idx):
        sample = list(self.data[idx]) #

        # x-ray image
        dcm_path1 = sample[0]
        dcm_path2 = sample[self.offset]

        array_path1 = os.path.join(self.data_dir,dcm_path1.replace('.dcm','.npy'))
        array_path2 = os.path.join(self.data_dir,dcm_path2.replace('.dcm','.npy'))

        image1 = np.load(array_path1)
        image1 = image1 - image1.min()
        image1 = image1 / image1.max()

        image2 = np.load(array_path2)
        image2 = image2 - image2.min()
        image2 = image2 / image2.max()

        # make base sample dict
        ret=self._make_sample_dict(sample)
        # conver to tensor
        ret = self._convert_to_tensor(ret)

        # data augmentation
        transformed = self.transform({'image':image1})
        image1 = transformed['image']
        transformed = self.transform({'image':image2})
        image2 = transformed['image']

        x1 = torch.cat([image1, image1, image1], dim = 0).to(torch.float32)
        x2 = torch.cat([image2, image2, image2], dim = 0).to(torch.float32)
        x = torch.cat([x1, x2], dim=0)

        ret.update({
            'image' : x, # image has size of (2 * C x H x V)
            'f_name1' : array_path1,
            'f_name2' : array_path2
        })

        return ret
    
if __name__ == '__main__':
    
    pass