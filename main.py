import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from warmup_scheduler import GradualWarmupScheduler

import albumentations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from dataset_rua import PANDADataset_Train
from model_rua import enetv2



data_dir = '/mnt/chengyao/prostate-cancer-grade-assessment'
df_train = pd.read_csv(os.path.join(data_dir, '../train5.csv'))
image_folder = os.path.join(data_dir, 'train_images')
gpu_mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 7
}


batch_size = 2
num_workers = 6
out_dim = 5
init_lr = 3e-4
warmup_factor = 10
warmup_epo = 1
n_epochs = 20


def split_folds():
    # Assign Folds
    skf = StratifiedKFold(5, shuffle = True, random_state = 42)
    df_train['fold'] = -1
    for i, (train_idx, valid_idx) in enumerate(skf.split(df_train, df_train['isup_grade'])):
        df_train.loc[valid_idx, 'fold'] = i
    df_train.to_csv(os.path.join(data_dir, '../train5.csv'), index = False)


def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    train_loss = []
    # for i, (data, target) in enumerate(loader):

    for i, (data, target) in tqdm(enumerate(loader), total = len(loader)):

        data, target = data.to(device), target.to(device)
        loss_func = criterion
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)

    return train_loss


def val_epoch(model, loader, device, criterion):

    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []

    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)

            loss = criterion(logits, target)

            pred = logits.sigmoid().sum(1).detach().round()
            LOGITS.append(logits)
            PREDS.append(pred)
            TARGETS.append(target.sum(1))

            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    acc = (PREDS == TARGETS).mean() * 100.
    
    qwk = cohen_kappa_score(PREDS, TARGETS, weights = 'quadratic')
    print(f'qwk: {qwk} from Device {device}')

    return val_loss, acc, qwk


def train(fold: int = 4):
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    device = torch.device(f'cuda:{gpu_mapping[fold]}')

    transforms_train = albumentations.Compose([
        albumentations.Transpose(p = 0.5),
        albumentations.VerticalFlip(p = 0.5),
        albumentations.HorizontalFlip(p = 0.5),
    ])
    transforms_val = albumentations.Compose([])

    criterion = nn.BCEWithLogitsLoss()

    df_this = df_train.loc[np.where((df_train['fold'] != fold))[0]]
    df_valid = df_train.loc[np.where((df_train['fold'] == fold))[0]]

    dataset_train = PANDADataset_Train(df_this, data_dir + '/train_images/', transform = transforms_train)
    dataset_valid = PANDADataset_Train(df_valid, data_dir + '/train_images/', transform = transforms_val)

    train_loader = DataLoader(dataset_train, batch_size = batch_size,
                              sampler = RandomSampler(dataset_train), num_workers = num_workers)
    valid_loader = DataLoader(dataset_valid, batch_size = batch_size,
                              sampler = SequentialSampler(dataset_valid), num_workers = num_workers)

    model = enetv2(backbone = 'efficientnet-b0',
                   backbone_pretrain = './efficientnet-b0-08094119.pth',
                   out_dim = out_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=init_lr / warmup_factor)
    scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, n_epochs-warmup_epo)
    scheduler = GradualWarmupScheduler(optimizer,
                                       multiplier = warmup_factor,
                                       total_epoch = warmup_epo,
                                       after_scheduler = scheduler_cosine)

    kernel_type = f'EfficientNet_B0_{fold}'
    qwk_max = 0.
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch} For Device: {device}')
        _ = train_epoch(model, train_loader, optimizer, device, criterion)
        scheduler.step(epoch - 1)
        _, acc, qwk = val_epoch(model, valid_loader, device, criterion)

        content = time.ctime() + ' ' + f'Epoch {epoch}|{device}|Acc: {(acc):.5f}|QWK: {(qwk):.5f}'
        with open(f'log_{kernel_type}.txt', 'a') as appender:
            appender.write(content + '\n')

        if qwk > qwk_max:
            torch.save(model.state_dict(), f'{kernel_type}_best_fold{fold}.pth')
            qwk_max = qwk

    torch.save(model.state_dict(), os.path.join(f'{kernel_type}_final_fold{fold}.pth'))



if __name__ == '__main__':


    # split_folds()

    import multiprocessing
    
    pool = [multiprocessing.Process(target = train, args = (i, )) for i in range(1)]
    for p in pool:
        p.start()
    for p in pool:
        p.join()
