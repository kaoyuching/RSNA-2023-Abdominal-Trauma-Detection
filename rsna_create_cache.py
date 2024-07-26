import os
import numpy as np
import pandas as pd
import cv2
import skimage
from tqdm.auto import tqdm

from rsna_folds import train_dict, valid_dict, MetaSchema, LabelSchema

from rsna_data_utils import read_volume
from rsna_data_utils import (
        calc_slice_stepv2, calc_mip, calc_minip, 
        calc_mid, scale_depth, convert_hu, get_mask_idx,
        find_height, find_width, resize_volume)


datadir = "/home/data4/share/rsna-2023-abdominal-trauma-detection/"
train_datadir = "/home/data4/share/rsna-2023-abdominal-trauma-detection/train_images/"
segment_datadir = "/home/doriskao/project/rsna_abdomen/predict_masks/size256_skimage_fold1/"
cache_path = "./cache/organ_data"


data_df = pd.read_csv('./train_new.csv')
label_df = data_df[[
    'patient_id', 'series_id',
    'bowel_injury', 'extravasation_injury',
    'kidney_healthy', 'kidney_low', 'kidney_high',
    'liver_healthy', 'liver_low', 'liver_high',
    'spleen_healthy', 'spleen_low', 'spleen_high',
    'any_injury',
    'SliceThickness', 'slice_spacing', 'number_slices', 'start_idx', 'end_idx',
]]

label_df['patient_id'] = label_df['patient_id'].astype(str)
label_df['series_id'] = label_df['series_id'].astype(str)
label_df['path'] = label_df.apply(lambda x: os.path.join(train_datadir, str(x['patient_id']), str(x['series_id'])), axis=1)
label_df['kidney'] = label_df.apply(lambda x: np.argmax(x[['kidney_healthy', 'kidney_low', 'kidney_high']]), axis=1)
label_df['liver'] = label_df.apply(lambda x: np.argmax(x[['liver_healthy', 'liver_low', 'liver_high']]), axis=1)
label_df['spleen'] = label_df.apply(lambda x: np.argmax(x[['spleen_healthy', 'spleen_low', 'spleen_high']]), axis=1)

label_df = label_df[['patient_id', 'series_id', 'path', 'bowel_injury', 'extravasation_injury', 'kidney', 'liver', 'spleen', 'any_injury', 'SliceThickness', 'slice_spacing', 'number_slices', 'start_idx', 'end_idx']]
label_df = label_df.rename(columns={'bowel_injury': 'bowel', 'extravasation_injury': 'extravasation', 'SliceThickness': 'slice_thickness'})
label_df['dummy_label'] = label_df.apply(lambda x: str(np.array(x[['bowel', 'extravasation', 'kidney', 'liver', 'spleen']])), axis=1)


label_dict = {i: {
    'path': s['path'],
    'mask': os.path.join(segment_datadir, str(s['series_id'])+'.npy'),
    'label': LabelSchema(bowel=s['bowel'], extravasation=s['extravasation'], kidney=s['kidney'], liver=s['liver'], spleen=s['spleen'], any_injury=s['any_injury']),
    'meta': MetaSchema(slice_thickness=s['slice_thickness'], slice_spacing=s['slice_spacing'], number_slices=s['number_slices'], start_idx=s['start_idx'], end_idx=s['end_idx'], series_id=s['series_id'])
} for i, s in label_df.iterrows()}


data_keys = list(label_dict.keys())
dsize = (256, 256)
max_depth = 64


for idx in tqdm(range(len(label_dict))):
    idx = data_keys[idx]
    data = label_dict[idx]
    fname = data['path']
    label = data['label']
    meta = data['meta']
    mask_path = data['mask']

    # load image and compute crop_h, crop_w
    raw_volume, _ = read_volume(fname, wl=None, ww=None, resize=dsize, square=False)  # (h, w, d)
    # find croped width and height
    _imgs = convert_hu(raw_volume, wl=150, ww=300)
    w_min, w_max = find_width(_imgs, px_thres=49, thres_ratio=7, extend_ratio=0.1, vol_ratio=0.8)
    h_min, h_max = find_height(_imgs, px_thres=33, thres_ratio=12, extend_ratio=0.1, vol_ratio=0.8)

    # load mask
    with open(mask_path, 'rb') as f:
        mask = np.load(f)  # (h, w, d)
    mask_w, mask_h = mask.shape[:2]
    mask = np.where(mask == 4, 0, mask)  # no bowel, organ only
    mask = skimage.transform.resize(
        mask,
        (mask_w, mask_h, meta.number_slices),
        anti_aliasing=False,
        mode='reflect',
        preserve_range=True,
        order=0
    )
    mask_start, mask_end = get_mask_idx(mask, px_threshold=50, extend_num=0, continuous_id=True)

    mask = mask[h_min:(h_max+1), w_min:(w_max+1), mask_start:(mask_end + 1)]  # (croph, cropw, cropd)
    _mask = [cv2.resize(mask[:, :, i], dsize, interpolation=cv2.INTER_NEAREST) for i in range(mask.shape[-1])]
    mask = np.stack(_mask, axis=2)

    # mask = np.where(mask == 4, 0, mask)  # no bowel
    # if mask.shape[-1] < self.max_depth:
        # mask = scale_depth(mask, self.max_depth, spacial=2, two_side=False)
    # else:
    mask = skimage.transform.resize(
        mask,
        (*dsize, max_depth),
        anti_aliasing=False,
        mode='reflect',
        preserve_range=True,
        order=0
    )
    mask = (mask/3*255).astype(np.uint8)

    # load image
    # volume, _ = read_volume(fname, wl=self.wl, ww=self.ww, resize=dsize, square=False)  # (h, w, d)
    volume = convert_hu(raw_volume, wl=50, ww=400)
    volume = volume[h_min:(h_max+1), w_min:(w_max+1), mask_start:(mask_end + 1)]  # (croph, cropw, cropd)
    volume = resize_volume(volume, resize=dsize)

    # volume = volume[:, :, mask_start:(mask_end + 1)]
    # if volume.shape[-1] < self.max_depth:
        # volume = scale_depth(volume, self.max_depth, spacial=2, two_side=False)
    # else:
    volume = skimage.transform.resize(volume, (*dsize, max_depth), anti_aliasing=False, preserve_range=True)  # (h, w, d)
    volume = np.stack([volume, mask], axis=0)  # (2, h, w, d)

    fname = os.path.basename(mask_path)
    cache_fpath = os.path.join(cache_path, fname)

    with open(cache_fpath, 'wb') as f:
        np.save(f, volume.astype(np.uint8))
