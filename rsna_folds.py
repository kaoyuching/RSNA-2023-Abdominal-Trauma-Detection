import os
import numpy as np
import pandas as pd
from pydantic import BaseModel
import dill

import warnings
warnings.filterwarnings("ignore")


datadir = "/home/data4/share/rsna-2023-abdominal-trauma-detection/"
train_datadir = "/home/data4/share/rsna-2023-abdominal-trauma-detection/train_images/"
# segment_datadir = "/home/doriskao/project/rsna_abdomen/predict_masks/size256_skimage/"
segment_datadir = "/home/doriskao/project/rsna_abdomen/predict_masks/size256_skimage_fold1/"
# segment_datadir = "/home/doriskao/project/rsna_abdomen/predict_masks/size256_skimage_fold1v2/"


class MetaSchema(BaseModel):
    slice_thickness: float
    slice_spacing: float
    pixel_spacing_i: float
    pixel_spacing_j: float
    number_slices: int
    start_idx: int
    end_idx: int
    series_id: int


class LabelSchema(BaseModel):
    bowel: int
    extravasation: int
    kidney: int
    liver: int
    spleen: int
    any_injury: int


data_df = pd.read_csv('./train_new.csv')
train_tag_df = data_df[data_df['train_tag']].reset_index(drop=True)

label_df = train_tag_df[[
    'patient_id', 'series_id',
    'bowel_injury', 'extravasation_injury',
    'kidney_healthy', 'kidney_low', 'kidney_high',
    'liver_healthy', 'liver_low', 'liver_high',
    'spleen_healthy', 'spleen_low', 'spleen_high',
    'any_injury',
    'SliceThickness', 'slice_spacing', 'number_slices', 'start_idx', 'end_idx',
    'pixel_spacing_i', 'pixel_spacing_j',
]]

label_df['patient_id'] = label_df['patient_id'].astype(str)
label_df['series_id'] = label_df['series_id'].astype(str)
label_df['path'] = label_df.apply(lambda x: os.path.join(train_datadir, str(x['patient_id']), str(x['series_id'])), axis=1)
label_df['kidney'] = label_df.apply(lambda x: np.argmax(x[['kidney_healthy', 'kidney_low', 'kidney_high']]), axis=1)
label_df['liver'] = label_df.apply(lambda x: np.argmax(x[['liver_healthy', 'liver_low', 'liver_high']]), axis=1)
label_df['spleen'] = label_df.apply(lambda x: np.argmax(x[['spleen_healthy', 'spleen_low', 'spleen_high']]), axis=1)

label_df = label_df[['patient_id', 'series_id', 'path', 'bowel_injury', 'extravasation_injury', 'kidney', 'liver', 'spleen', 'any_injury', 'SliceThickness', 'slice_spacing', 'number_slices', 'start_idx', 'end_idx', 'pixel_spacing_i', 'pixel_spacing_j']]
label_df = label_df.rename(columns={'bowel_injury': 'bowel', 'extravasation_injury': 'extravasation', 'SliceThickness': 'slice_thickness'})
label_df['dummy_label'] = label_df.apply(lambda x: str(np.array(x[['bowel', 'extravasation', 'kidney', 'liver', 'spleen']])), axis=1)


# create train / valid data
with open("/home/doriskao/project/rsna_abdomen/data/folds_info.pkl", 'rb') as f:
    folds_info = dill.load(f)

train_idx = folds_info['fold0']['train']
valid_idx = folds_info['fold0']['valid']

# invalid_series_id = [59, 26129, 31476, 23548, 8946]
invalid_series_id = []

label_dict = {i: {
    'path': s['path'],
    'mask': os.path.join(segment_datadir, str(s['series_id'])+'.npy'),
    'label': LabelSchema(bowel=s['bowel'], extravasation=s['extravasation'], kidney=s['kidney'], liver=s['liver'], spleen=s['spleen'], any_injury=s['any_injury']),
    'meta': MetaSchema(slice_thickness=s['slice_thickness'], slice_spacing=s['slice_spacing'], pixel_spacing_i=s['pixel_spacing_i'], pixel_spacing_j=s['pixel_spacing_j'], number_slices=s['number_slices'], start_idx=s['start_idx'], end_idx=s['end_idx'], series_id=s['series_id'])
} for i, s in label_df.iterrows() if s['series_id'] not in invalid_series_id}

train_dict = {k: v for k, v in label_dict.items() if k in train_idx}
valid_dict = {k: v for k, v in label_dict.items() if k in valid_idx}
