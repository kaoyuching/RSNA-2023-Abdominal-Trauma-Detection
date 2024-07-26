import os
import numpy as np
import cv2
import skimage
import torch
from torch.utils.data import Dataset, DataLoader

from rsna_data_utils import read_volume
from rsna_data_utils import (
        calc_slice_stepv2, calc_mip, calc_minip, 
        calc_mid, scale_depth, convert_hu, get_mask_idx,
        find_height, find_width, resize_volume, find_mask_margin)


class TraumaDataset(Dataset):
    def __init__(
        self,
        data_dict,
        resize_h=None,
        resize_w=None,
        wl=200,
        ww=400,
        max_depth=64,
        transforms=None,
        mode: str = 'resize',
        cache_path = None,
    ):
        self.data_dict = data_dict
        self.data_keys = list(data_dict.keys())
        self.wl = wl
        self.ww = ww
        self.max_depth = max_depth
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.transforms = transforms
        self.mode = mode
        self.cache_path = cache_path

    def __len__(self):
        return len(self.data_keys)

    def get_random_weights(self, supress_all_negtive: float = 1, draw: bool = False):
        _all_negative = (np.array([v['label'].any_injury for k, v in self.data_dict.items()]) == 0)
        w_neg = (1-_all_negative.mean()) * supress_all_negtive
        sws = _all_negative*w_neg + (1-_all_negative)*1
        if draw:
            plt.hist(sws, weights=sws)
            plt.show()
        return sws

    def __getitem__(self, idx):
        idx = self.data_keys[idx]
        data = self.data_dict[idx]
        fname = data['path']
        label = data['label']
        meta = data['meta']
        mask_path = data['mask']

        if self.resize_h is not None and self.resize_w is not None:
            dsize = (self.resize_w, self.resize_h)

        cache_fname = ''
        if self.cache_path is not None:
            cache_fname = os.path.join(self.cache_path, str(meta.series_id)+'.npy')

        if os.path.exists(cache_fname):
            with open(cache_fname, 'rb') as f:
                volume = np.load(f)  # (2, h, w, d)
        else:
            # load image and compute crop_h, crop_w
            # raw_volume, _ = read_volume(fname, wl=None, ww=None, resize=dsize, square=False)  # (h, w, d)
            raw_volume, _ = read_volume(
                fname, 
                wl=None, 
                ww=None, 
                resize=None, 
                square=False, 
                spacing=(meta.pixel_spacing_i, meta.pixel_spacing_j),
                new_spacing=1.0,
            )  # (h, w, d)

            # find croped width and height
            _imgs = convert_hu(raw_volume, wl=150, ww=300)
            w_min, w_max = find_width(_imgs, px_thres=49, thres_ratio=7, extend_ratio=0.1, vol_ratio=0.8)
            h_min, h_max = find_height(_imgs, px_thres=33, thres_ratio=12, extend_ratio=0.1, vol_ratio=0.8)

            # load mask
            with open(mask_path, 'rb') as f:
                mask = np.load(f)  # (h, w, d)
            mask_w, mask_h = mask.shape[:2]
            # mask = np.where(mask == 4, 0, mask)  # no bowel, organ only
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

            mask = np.where(mask == 4, 0, mask)  # no bowel
            mask = skimage.transform.resize(
                mask,
                (*dsize, self.max_depth),
                anti_aliasing=False,
                mode='reflect',
                preserve_range=True,
                order=0
            )
            mask = (mask/3*255).astype(np.uint8)

            # load image
            volume = convert_hu(raw_volume, wl=self.wl, ww=self.ww)
            volume = volume[h_min:(h_max+1), w_min:(w_max+1), mask_start:(mask_end + 1)]  # (croph, cropw, cropd)
            volume = resize_volume(volume, resize=dsize)

            if self.mode == 'resize':
                volume = skimage.transform.resize(volume, (*dsize, self.max_depth), anti_aliasing=False, preserve_range=True)  # (h, w, d)
                # volume = np.stack([volume, mask], axis=0)  # (2, h, w, d)
                volume = np.expand_dims(volume, axis=0)  # (1, h, w, d)
            else:
                step = calc_slice_stepv2(
                    meta.slice_thickness,
                    meta.slice_spacing,
                    10,
                    meta.number_slices,
                    target_slices=self.max_depth,
                    default_step=1
                )
                mip_img = calc_mip(volume, slices_num=int(step))  # maximum pooling
                minip_img = calc_minip(volume, slices_num=int(step))  # minimum pooling
                mid_img = calc_mid(volume, slices_num=int(step))  # middle
                volume = np.stack([mip_img, mid_img, minip_img], axis=0)  # (c, h, w, d)
                # align volume depth
                volume = scale_depth(volume, self.max_depth, spacial=3)  # (c, h, w, d)
                
                # add mask (h, w, d) -> (1, h, w, d)
        #         volume = np.concatenate([volume, np.expand_dims(mask, axis=0)], axis=0)
                del mip_img, minip_img, mid_img

        #transformer
        if self.transforms is not None:
            aug = self.transforms({'image': volume})
            volume = aug['image'].numpy()

        volume = volume / 255.  # (C, H, W, D)
        volume = torch.from_numpy(volume)
        volume = volume.permute(0, 3, 1, 2)  # (C, H, W, D) -> (C, D, H, W)
        return volume, label


# {'background': 0, 'liver': 1, 'spleen': 2, 'kidney': 3, 'bowel': 4}
class TraumaDatasetV2(Dataset):
    def __init__(
        self,
        data_dict,
        resize_h=None,
        resize_w=None,
        wl=200,
        ww=400,
        max_depth=64,
        transforms=None,
        mode: str = 'resize',
        cache_path = None,
    ):
        self.data_dict = data_dict
        self.data_keys = list(data_dict.keys())
        self.wl = wl
        self.ww = ww
        self.max_depth = max_depth
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.transforms = transforms
        self.mode = mode
        self.cache_path = cache_path

    def __len__(self):
        return len(self.data_keys)

    def get_random_weights(self, supress_all_negtive: float = 1, draw: bool = False):
        _all_negative = (np.array([v['label'].any_injury for k, v in self.data_dict.items()]) == 0)
        w_neg = (1-_all_negative.mean()) * supress_all_negtive
        sws = _all_negative*w_neg + (1-_all_negative)*1
        if draw:
            plt.hist(sws, weights=sws)
            plt.show()
        return sws

    def _crop_organ(self, img, mask, organ_id, default_x=(None, None), default_y=(None, None)):
        # mask: (h, w, d)
        xmin, xmax = find_mask_margin(mask, organ_id, axis=1, extend_num=5, default_start=default_x[0], default_end=default_x[1])
        ymin, ymax = find_mask_margin(mask, organ_id, axis=0, extend_num=5, default_start=default_y[0], default_end=default_y[1])
        zmin, zmax = find_mask_margin(mask, organ_id, axis=2, extend_num=1)
        crop_organ = img[ymin:(ymax+1), xmin:(xmax+1), zmin:(zmax+1)]
        dsize = (self.resize_w, self.resize_h)
        _img = [cv2.resize(crop_organ[:, :, i], dsize, interpolation=cv2.INTER_AREA) for i in range(crop_organ.shape[-1])]
        crop_organ = np.stack(_img, axis=2)
        crop_organ = skimage.transform.resize(crop_organ, (*dsize, self.max_depth), anti_aliasing=False, preserve_range=True)  # (h, w, d)
        return crop_organ

    def __getitem__(self, idx):
        idx = self.data_keys[idx]
        data = self.data_dict[idx]
        fname = data['path']
        label = data['label']
        meta = data['meta']
        mask_path = data['mask']

        if self.resize_h is not None and self.resize_w is not None:
            dsize = (self.resize_w, self.resize_h)

        cache_fname = ''
        if self.cache_path is not None:
            cache_fname = os.path.join(self.cache_path, str(meta.series_id)+'.npy')

        if os.path.exists(cache_fname):
            with open(cache_fname, 'rb') as f:
                volume = np.load(f)  # (2, h, w, d)
        else:
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
            mask = mask[:, :, mask_start:(mask_end + 1)]  # (croph, cropw, cropd)


            # load image
            volume = convert_hu(raw_volume, wl=self.wl, ww=self.ww)
            organ_volume = volume[:, :, mask_start:(mask_end + 1)]  # (croph, cropw, cropd)
            # full
            volume = volume[h_min:(h_max+1), w_min:(w_max+1), mask_start:(mask_end + 1)]  # (croph, cropw, cropd)
            volume = resize_volume(volume, resize=dsize)

            volume = skimage.transform.resize(volume, (*dsize, self.max_depth), anti_aliasing=False, preserve_range=True)  # (h, w, d)
            # volume = np.stack([volume, mask], axis=0)  # (2, h, w, d)
            volume = np.expand_dims(volume, axis=0)  # (1, h, w, d)
            # crop organ
            crop_kidney = self._crop_organ(organ_volume, mask, 3, default_x=(w_min, w_max+1), default_y=(h_min, h_max+1))
            crop_kidney = np.expand_dims(crop_kidney, axis=0)  # (1, h, w, d)
            crop_liver = self._crop_organ(organ_volume, mask, 1, default_x=(w_min, w_max+1), default_y=(h_min, h_max+1))
            crop_liver = np.expand_dims(crop_liver, axis=0)  # (1, h, w, d)
            crop_spleen = self._crop_organ(organ_volume, mask, 2, default_x=(w_min, w_max+1), default_y=(h_min, h_max+1))
            crop_spleen = np.expand_dims(crop_spleen, axis=0)  # (1, h, w, d)

        #transformer
        if self.transforms is not None:
            aug = self.transforms({'image': volume})
            volume = aug['image'].numpy()
            kidney_aug = self.transforms({'image': crop_kidney})
            crop_kidney = kidney_aug['image'].numpy()
            liver_aug = self.transforms({'image': crop_liver})
            crop_liver = liver_aug['image'].numpy()
            spleen_aug = self.transforms({'image': crop_spleen})
            crop_spleen = spleen_aug['image'].numpy()

        volume = volume / 255.  # (C, H, W, D)
        volume = torch.from_numpy(volume)
        volume = volume.permute(0, 3, 1, 2).squeeze(0)  # (C, H, W, D) -> (C, D, H, W)
        crop_kidney = crop_kidney / 255.  # (C, H, W, D)
        crop_kidney = torch.from_numpy(crop_kidney)
        crop_kidney = crop_kidney.permute(0, 3, 1, 2).squeeze(0)  # (C, H, W, D) -> (C, D, H, W)
        crop_liver = crop_liver / 255.  # (C, H, W, D)
        crop_liver = torch.from_numpy(crop_liver)
        crop_liver = crop_liver.permute(0, 3, 1, 2).squeeze(0)  # (C, H, W, D) -> (C, D, H, W)
        crop_spleen = crop_spleen / 255.  # (C, H, W, D)
        crop_spleen = torch.from_numpy(crop_spleen)
        crop_spleen = crop_spleen.permute(0, 3, 1, 2).squeeze(0)  # (C, H, W, D) -> (C, D, H, W)
        return volume, crop_kidney, crop_liver, crop_spleen, label


# collate function
def collate_fn(batch):
    # batch: [(<image tensor (c,d,h,w)>, <labels {task: label}>)]
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    label = {
        'bowel': torch.unsqueeze(torch.Tensor([x.bowel for x in labels]), dim=1),
        'extravasation': torch.unsqueeze(torch.Tensor([x.extravasation for x in labels]), dim=1),
        'kidney': torch.unsqueeze(torch.Tensor([x.kidney for x in labels]), dim=1),
        'liver': torch.unsqueeze(torch.Tensor([x.liver for x in labels]), dim=1),
        'spleen': torch.unsqueeze(torch.Tensor([x.spleen for x in labels]), dim=1),
        'any_injury': torch.unsqueeze(torch.Tensor([x.any_injury for x in labels]), dim=1)
    }
    return imgs, label


def collate_fn_v2(batch):
    # batch: [(<image tensor (c,d,h,w)>, <labels {task: label}>)]
    imgs, crop_kidney, crop_liver, crop_spleen, labels = zip(*batch)
    imgs = torch.stack(imgs)
    crop_kidney = torch.stack(crop_kidney)
    crop_liver = torch.stack(crop_liver)
    crop_spleen = torch.stack(crop_spleen)
    label = {
        'bowel': torch.unsqueeze(torch.Tensor([x.bowel for x in labels]), dim=1),
        'extravasation': torch.unsqueeze(torch.Tensor([x.extravasation for x in labels]), dim=1),
        'kidney': torch.unsqueeze(torch.Tensor([x.kidney for x in labels]), dim=1),
        'liver': torch.unsqueeze(torch.Tensor([x.liver for x in labels]), dim=1),
        'spleen': torch.unsqueeze(torch.Tensor([x.spleen for x in labels]), dim=1),
        'any_injury': torch.unsqueeze(torch.Tensor([x.any_injury for x in labels]), dim=1)
    }
    return imgs, crop_kidney, crop_liver, crop_kidney, label
