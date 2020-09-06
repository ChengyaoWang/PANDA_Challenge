'''
    Filtering out Noisy Labels
'''
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import model as enet

from dataset import PANDADataset_Inference


def clear_add_pred_col():
    df_train = pd.read_csv('/mnt/chengyao/train5.csv')
    dataset_size = len(df_train.index)
    df_train['pred'] = [float('-inf') for _ in range(dataset_size)]
    
    for idx in range(dataset_size):
        if df_train.iloc[idx].image_id == '3790f55cad63053e956fb73027179707':
            print(f'Droping Blank Slide {idx} with ID = 3790f55cad63053e956fb73027179707')
            df_train = df_train.drop([idx]).reset_index(drop = True)
            break
    df_train.to_csv('/mnt/chengyao/train5.csv', index = False)


def load_model(model_f):
    model_f = os.path.join(model_dir, model_f)
    backbone = 'efficientnet-b0'
    model = enetv2(backbone, out_dim = 5)
    model.load_state_dict(torch.load(model_f, map_location = lambda storage, loc: storage), strict = True)
    model.eval()
    print(f'{model_f} loaded!')
    return model

class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x

def predict_one_fold(fold: int, mpLock = None):

    model = load_model(f'EfficientNet_B0_{fold}_best_fold{fold}.pth')
    device = torch.device(f'cuda:{gpu_mapping[fold]}')
    model.to(device)

    df_train = pd.read_csv('/mnt/chengyao/train5.csv')
    df_train_fold = df_train.loc[np.where((df_train['fold'] == fold))[0]].reset_index(drop = True)

    dataset = PANDADataset_Inference(df_train_fold, image_folder)
    loader = DataLoader(dataset, batch_size = 8, num_workers = 4, shuffle = False)

    LOGITS1 = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            print(f'{i} / {len(loader)} CUDA:{fold}')
            data = data.to(device)
            LOGITS1.append(model(data))

    LOGITS = torch.cat(LOGITS1).sigmoid().cpu()
    PREDS = LOGITS.sum(1).numpy()

    df_train = pd.read_csv('/mnt/chengyao/train5.csv')
    df_train.loc[np.where((df_train['fold'] == fold))[0], 'pred'] = PREDS.astype(float)

    mpLock.acquire()    
    df_train.to_csv('/mnt/chengyao/train5.csv', index = False)
    mpLock.release()

    print(f'Fold {fold} Complete')


# Show Distribution over Different Institutions
def filter_samples(threshold: float = 1.5):
    institution_distribution = {'karolinska': 0, 'radboud': 0}
    data_csv = pd.read_csv('/mnt/chengyao/train5.csv')
    newCSV_Idx = []
    for rowIdx in range(len(data_csv.index)):
        diff = abs(data_csv.loc[rowIdx, 'pred'] - data_csv.loc[rowIdx, 'isup_grade'])
        if diff >= threshold:
            institution_distribution[data_csv.loc[rowIdx, 'data_provider']] += 1
        else:
            newCSV_Idx.append(rowIdx)
    newCSV_Idx = np.array(newCSV_Idx)
    print(f'New Size of Dataset: {newCSV_Idx.shape[0]}')
    data_csv = data_csv.loc[newCSV_Idx].reset_index(drop = True)
    data_csv.to_csv('/mnt/chengyao/prostate-cancer-grade-assessment/train5.csv')
    print(institution_distribution)
    # {'karolinska': 254, 'radboud': 567}    


if __name__ == '__main__':
    
    model_dir = './'
    gpu_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 7}
    image_folder = '/mnt/chengyao/prostate-cancer-grade-assessment/train_images/'

    clear_add_pred_col()

    import multiprocessing
    mpLock = multiprocessing.Lock()

    pool = [multiprocessing.Process(target = predict_one_fold, args = (i, mpLock, )) for i in range(5)]
    for p in pool:
        p.start()
    for p in pool:
        p.join()

    # filter_samples()
