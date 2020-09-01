import math, cv2
import numpy as np 
import torch
from torch.utils.data import Dataset
from optimal_tiling import opt_tiling


npy_folder = '/mnt/chengyao/prostate-cancer-grade-assessment/npy/'

class PANDADataset_Train(Dataset):
    def __init__(self, df, data_path, transform = None, task = 'clf'):
        self.df = df.reset_index(drop = True)
        self.transform = transform
        self.extractor = opt_tiling(df, data_path)
        self.task = task

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = self.df.iloc[index].image_id
        stacked_patches = np.load(npy_folder + img_id + '.npy')

        l = int(math.sqrt(self.extractor.sample_size))
        retImg = np.empty(
            (self.extractor.sample_size, self.extractor.patch_size, self.extractor.patch_size, 3), dtype = int
        )
        for i, this_img in enumerate(stacked_patches):
            if self.transform is not None:
                this_img = self.transform(image = this_img)['image']
            retImg[i] = this_img
        retImg = cv2.hconcat([
            cv2.vconcat([retImg[v + h * l] for v in range(l)]) for h in range(l)
        ])

        if self.transform is not None:
            retImg = self.transform(image=retImg)['image']
        retImg = 1. - retImg.astype(np.float32) / 255.
        retImg = retImg.transpose(2, 0, 1)

        if self.task == 'clf':
            label = np.zeros(5).astype(np.float32)
            label[:row.isup_grade] = 1.

        elif self.task == 'reg':
            label = [row.isup_grade]

        else:
            raise ValueError('Please Use Specify Task Type')

        return torch.tensor(retImg), torch.tensor(label, dtype = torch.float32)



class PANDADataset_Inference(Dataset):

    def __init__(self, df, data_path):
        self.df = df.reset_index(drop = True)
        self.extractor = opt_tiling(df, data_path)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        img = self.extractor.read_image(index)
        img, coords = self.extractor.locate_tiles(img)
        concate_patches = self.extractor.yield_concat_patches(img, coords, True)

        concate_patches = 1. - concate_patches.astype(np.float32) / 255.
        concate_patches = concate_patches.transpose(2, 0, 1)

        return torch.tensor(concate_patches)