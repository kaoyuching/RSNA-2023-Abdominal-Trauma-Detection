import os
import logging
import numpy as np
import pandas as pd
import cv2
import yaml
import json
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch import optim

from monai.transforms import Compose, Resized
import monai.transforms as transforms

from rsna_data_utils import read_volume, calc_slice_stepv2, calc_mip, calc_minip, calc_mid, scale_depth, set_seed, convert_hu
from rsna_folds import train_dict, valid_dict
from rsna_torch_data import TraumaDataset, collate_fn, TraumaDatasetV2, collate_fn_v2
from rsna_model_utils import RSNAModelTimm, RSNAModelTimmOrgan, Custom3DCNN, MILClassificationModel
from rsna_utils import custom_metric, custom_loss, custom_loss_organ, custom_metric_organ, custom_loss_v2
from rsna_training_utils import run_train, run_valid, run_valid_organ, run_train_v2, run_valid_v2

import warnings
warnings.filterwarnings("ignore")


# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
set_seed(42)
train_config_path = 'train_config.yml'

# load config
with open(train_config_path, 'r') as f:
    train_config = yaml.safe_load(f)

wl = train_config.get('window_level', 50)
ww = train_config.get('window_width', 400)
h = train_config.get('height', 256)
w = train_config.get('width', 256)
max_depth = train_config['max_depth']
batch_size = train_config.get('batch_size', 4)
num_workers = train_config.get('num_workers', 4)
cache_path = train_config.get('cache_path', None)
print(cache_path)

# logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(train_config.get('log_filename', './log_record/train_log.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logging.info(json.dumps(train_config))


# transform
train_transform = transforms.Compose([
    transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
    transforms.RandFlipd(keys=["image"], prob=0.25, spatial_axis=0),
    transforms.RandRotated(keys=["image"], range_x=0, range_y=0, range_z=(-7*np.pi/180, 7*np.pi/180), padding_mode='zeros', mode='nearest', prob=0.5),
    # transforms.RandAffined(keys=["image"], translate_range=[int(x*y) for x, y in zip([h, w], [0.3, 0.3, 0.3])], padding_mode='zeros', prob=0.3),
    transforms.RandGridDistortiond(keys=["image"], distort_limit=(-0.01, 0.01), mode="nearest", prob=0.3),
    transforms.RandAdjustContrastd(keys=["image"], gamma=(0.65, 2), prob=0.5),
    # transforms.RandZoomd(keys=["image"], min_zoom=[0.85, 1], max_zoom=[1, 1], padding_mode='empty', mode='nearest', prob=0.3),
    # transforms.RandGridDistortiond(keys=["image"], num_cells=4, distort_limit=(-0.2, 0.2), padding_mode="zeros", mode='nearest', prob=0.25),
    # transforms.RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=15),
])


# dataset
logging.info('create dataset...')
dataset_mode = train_config.get('dataset_mode', 'resize')
train_dataset = TraumaDataset(train_dict, resize_h=h, resize_w=w, wl=wl, ww=ww, max_depth=max_depth, transforms=train_transform, mode=dataset_mode, cache_path=cache_path)
valid_dataset = TraumaDataset(valid_dict, resize_h=h, resize_w=w, wl=wl, ww=ww, max_depth=max_depth, mode=dataset_mode, cache_path=cache_path)
# train_dataset = TraumaDatasetV2(train_dict, resize_h=h, resize_w=w, wl=wl, ww=ww, max_depth=max_depth, transforms=train_transform, mode=dataset_mode, cache_path=cache_path)
# valid_dataset = TraumaDatasetV2(valid_dict, resize_h=h, resize_w=w, wl=wl, ww=ww, max_depth=max_depth, mode=dataset_mode, cache_path=cache_path)


# sampler
sampler = None
# sampler = WeightedRandomSampler(
    # train_dataset.get_random_weights(supress_all_negtive=0.2),
    # len(train_dataset),
    # replacement=True,
# )

# dataloader
logging.info('create dataloader...')
# collate_fn = collate_fn_v2
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(sampler is None), collate_fn=collate_fn, num_workers=num_workers, drop_last=True, sampler=sampler)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

# model
model_name = train_config.get('model_name', 'resnet50')
model_pretrained = train_config.get('model_pretrained', None)
organ_only = train_config.get('organ_only', False)
logging.info(f"load model {model_name}, organ only: {organ_only}")
if organ_only:
    model = RSNAModelTimmOrgan(
        model_name,
        pretrained=model_pretrained,
        n_input_channels=2,
    )
else:
    # model = RSNAModelTimm(
        # model_name,
        # pretrained=model_pretrained,
        # n_input_channels=2,
    # )
    # model = Custom3DCNN(
        # hidden=368,
        # num_channel=2,
    # )
    model = MILClassificationModel(
        model_name, 
        True,
        {'in_chans': 1, 'drop_rate': 0.1, 'drop_path_rate': 0.1},
        'avg', 
        'attention', 
        0.0,
        False
    )

model = model.to(device)


# hyperparameter setting
learning_rate = train_config['learning_rate']
weight_decay = train_config['weight_decay']

if organ_only:
    loss_fn = partial(custom_loss_organ, reduction='sum')
else:
    # loss_fn = partial(custom_loss, any_injury=False, reduction='mean')
    loss_fn = partial(custom_loss, any_injury=True, reduction='mean')
    # loss_fn = partial(custom_loss_v2, any_injury=True, reduction='mean')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-7, weight_decay=weight_decay)

num_train_steps = len(train_dataloader)
epochs = train_config.get('epochs', 20)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_train_steps//2, T_mult=1, eta_min=train_config.get('eta_min', 2.0e-6))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2500, eta_min=train_config.get('eta_min', 2.5e-5))
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=train_config.get('eta_min', 2.0e-6))

model_path_loss = train_config.get('model_path_loss', './models/model_loss.pth')
model_path_score = train_config.get('model_path_score', './models/model_score.pth')

# start training
logging.info('start training...')
valid_true_labels = {
    'bowel': torch.tensor([[x['label'].bowel] for x in list(valid_dict.values())]),
    'extravasation': torch.tensor([[x['label'].extravasation] for x in list(valid_dict.values())]),
    'kidney': torch.tensor([[x['label'].kidney] for x in list(valid_dict.values())]),
    'liver': torch.tensor([[x['label'].liver] for x in list(valid_dict.values())]),
    'spleen': torch.tensor([[x['label'].spleen] for x in list(valid_dict.values())]),
    'any_injury': torch.tensor([[x['label'].any_injury] for x in list(valid_dict.values())]),
}

# epochs = train_config.get('epochs', 20)
best_loss = np.Inf
best_score = np.Inf

for epoch in range(epochs):
    # print(f'epoch {epoch + 1}/{epochs}')
    logging.info(f'epoch {epoch + 1}/{epochs}')

    train_loss = run_train(train_dataloader, model, loss_fn, optimizer, scheduler, device)
    # train_loss = run_train_v2(train_dataloader, model, loss_fn, optimizer, scheduler, device)
    if organ_only:
        valid_loss, valid_score, scores = run_valid_organ(valid_dataloader, model, loss_fn, device, valid_true_labels)
    else:
        valid_loss, valid_score, scores = run_valid(valid_dataloader, model, loss_fn, device, valid_true_labels)
        # valid_loss, valid_score, scores = run_valid_v2(valid_dataloader, model, loss_fn, device, valid_true_labels)

    if valid_loss < best_loss:
        logging.info("saving model(loss)...")
        torch.save(model.state_dict(), model_path_loss)
        best_loss = valid_loss

    if valid_score < best_score:
        logging.info("saving model(score)...")
        torch.save(model.state_dict(), model_path_score)
        best_score = valid_score


    if organ_only:
        logging.info(
            f'train loss: {train_loss} valid loss: {valid_loss} valid score: {valid_score}\n \
            kidney: {scores[0]} liver: {scores[1]} spleen: {scores[2]}'
        )
    else:
        logging.info(
            f'train loss: {train_loss} valid loss: {valid_loss} valid score: {valid_score}\n \
            bowel: {scores[0]} ext: {scores[1]} kidney: {scores[2]} liver: {scores[3]} spleen: {scores[4]} any: {scores[5]}'
        )

    # scheduler.step()
