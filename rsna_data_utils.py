import os
import numpy as np
import random
import cv2
import skimage
import pydicom
import dcmtrans
from dcmtrans import collect_dicoms, reconstruct_series
import nibabel as nib
import torch


def set_seed(seed):
    torch.manual_seed(seed) # cpu
    np.random.seed(seed) #numpy
    random.seed(seed) #random and transforms

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) #gpu
        torch.backends.cudnn.deterministic=True # cudnn


def convert_hu(hu_img: np.ndarray, wl: int, ww: int):
    ymin, ymax = 0, 255
    lb = wl - 0.5 - (ww - 1)/2
    ub = wl - 0.5 + (ww - 1)/2
    
    img = np.piecewise(
        hu_img,
        [hu_img <= lb, hu_img > ub, (hu_img > lb)&(hu_img <= ub)],
        [lambda x: ymin, lambda x: ymax, lambda x: ((x - (wl - 0.5))/(ww - 1) + 0.5)*(ymax - ymin) + ymin]
    )
    img = img.astype(np.uint8)
    return img


def pad_to_square(volumes, spacial=2):
    if spacial == 3:
        h, w, d = volumes.shape
    elif spacial == 2:
        h, w = volumes.shape
    else:
        raise(ValueError("Invalid spacial number. It should be 2 or 3"))

    if w > h:
        pad = w - h
        pad0 = pad // 2
        pad1 = pad - pad0
        
        if spacial == 3:
            npad = ((pad0, pad1), (0, 0), (0, 0))
        else:
            npad = ((pad0, pad1), (0, 0))
        
        volumes = np.pad(volumes, pad_width=npad, mode='constant', constant_values=0)

    if w < h:
        pad = h - w
        pad0 = pad // 2
        pad1 = pad - pad0

        if spacial == 3:
            npad = ((0, 0), (pad0, pad1), (0, 0))
        else:
            npad = ((0, 0), (pad0, pad1))
        volumes = np.pad(volumes, pad_width=npad, mode='constant', constant_values=0)
    return volumes

        
def read_nii(path=None, remove_bowel=False, resize=None, mask=False, nib_obj=None, square=False):
    if nib_obj is not None:
        vol = nib_obj
    else:
        vol = nib.load(path).dataobj
    vol = np.rot90(vol[:, :, ::-1], axes=(0, 1))
    if remove_bowel:
        vol = np.where(vol == 5, 0, vol)
    # np.flip(vol[..., i].T)
    if square:
        vol = pad_to_square(vol, spacial=3)
    
    if resize is not None:
        if mask:
            vol = np.stack([cv2.resize(vol[:, :, i], resize, interpolation=cv2.INTER_NEAREST) for i in range(vol.shape[-1])], axis=2)
            vol = np.round(vol, decimals=0)
        else:
            vol = np.stack([cv2.resize(vol[:, :, i], resize) for i in range(vol.shape[-1])], axis=2)
    return vol


def read_dicom(fname: str, wl: int = None, ww: int = None, reraise_modality: bool = False):
    dcmobj = pydicom.dcmread(fname, force=True)

    # read PixelData: return None when all methods are fail
    img = dcmtrans.read_pixel(fname, return_on_fail=None)

    # apply modality, value of interest, photometric interpretation transform to row image (np.array)
    window = None if wl is None or ww is None else [{'window_center': wl, 'window_width': ww}]
    imgs, excepts, info = dcmtrans.dcmtrans(dcmobj, img, window=window, depth=256)
    for e in excepts:
        if e is not None:
            raise e
    if reraise_modality and info['modality'] is None:
        raise ValueError('Cannot read modality transform attributes.')
    return imgs[0]


def get_dcmfiles(ppath):
    r'''
    image_mapping: {idx: fpath}
    '''
    dcm_files = []
    for root, dirs, files in os.walk(ppath):
        if files is not None:
            for f in files:
                dcm_files.append(os.path.join(root, f))
                
    collections = collect_dicoms(dcm_files, verbose=False)
    for k, v in collections.items():
        res = reconstruct_series(v, modality='CT', ignore_index_jump=True)
        
    image_mapping = {i: res.index_map[k] for i, k in enumerate(res.index_list)}
    return dcm_files, image_mapping, res.index_list, res.chirality



def resize_by_spacing(volumes, spacing: tuple, max_size: int = 256):
    r'''
    volumes: (h, w, d)
    spacing: (slice spacing, pixel spacing j, pixel spacing i)
    '''
    dz, dy, dx = spacing
    s, s, d = volumes.shape # scale to max size

    if max_size != s:
        scale = max_size / s

        nd = int(dz / dy * d * 0.5)  # we use sapcing dz,dy,dx = 2,1,1
        nd = int(scale * nd)
        h = int(scale * s)
        w = int(scale * s)

        volumes = np.stack([cv2.resize(volumes[:, :, i], (h, w), interpolation=cv2.INTER_LINEAR) for i in range(d)], axis=2)
        volumes = skimage.transform.resize(volumes, (h, w, nd), anti_aliasing=False, preserve_range=True)
    return volumes


def adjust_pixel_spacing(w, h, spacing: tuple = None, new_px_spacing=1.):
    r'''
    - volume: (h, w, d)
    - spacing: (spacing w, spacing h)
    '''
    if not isinstance(spacing, tuple):
        raise TypeError("spacing should be type tuple")
    px_w, px_h = spacing  # spacing w, spacing h
    px_rw, px_rh = px_w / new_px_spacing, px_h / new_px_spacing
    # h, w = volume.shape[:2]
    new_w, new_h = np.int(np.round(w*px_rw)), np.int(np.round(h*px_rh))
    return new_w, new_h


def read_volume(fpath, wl=None, ww=None, resize=None, square: bool = False, spacing: tuple = None, new_spacing: float = None):
    dcm_files, image_mapping, _, chirality = get_dcmfiles(fpath)

    imgs = []
    for k, v in image_mapping.items():
        img = read_dicom(v, wl=wl, ww=ww)
        h, w = img.shape

        if (spacing is not None) and (new_spacing is not None):
            adj_w, adj_h = adjust_pixel_spacing(w, h, spacing, new_spacing)
            img = cv2.resize(img, (adj_w, adj_h), interpolation=cv2.INTER_AREA)
        if square:
            img = pad_to_square(img, spacial=2)
        if resize is not None:  # resize (col, row)<(w, h)>
            img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
        imgs.append(img)
        
    imgs = np.stack(imgs, axis=2)
    return imgs, chirality


def read_volumev2(fpath, wl=None, ww=None, resize: int = None, spacing: tuple = None):
    r'''
    spacing: (slice spacing, pixel spacing j, pixel spacing i)
    '''
    dcm_files, image_mapping, *_ = get_dcmfiles(fpath)

    imgs = []
    for k, v in image_mapping.items():
        img = read_dicom(v, wl=wl, ww=ww)
        imgs.append(img)
    imgs = np.stack(imgs, axis=2)
    
    imgs = pad_to_square(imgs, spacial=3)  # (h, w, d)
    if resize is not None:
        imgs = resize_by_spacing(imgs, spacing=spacing, max_size=resize)
    return imgs


# compute slice v1
def calc_slice_step(thickness, spacing, exp_thickness):
    r"""
    exp_thickness = step * thickness - (thickness - spacing) * (step - 1)
    
    step is number slices to merge
    """
    step = (exp_thickness + spacing - thickness) / spacing
    return np.ceil(step)


# compute slice v2
def calc_slice_stepv2(thickness, spacing, exp_thickness, num_slices, target_slices=64, default_step=1):
    r"""
    exp_thickness = step * thickness - (thickness - spacing) * (step - 1)
    
    step is number slices to merge
    """
    if num_slices <= target_slices:
        return default_step
    else:
        step = (exp_thickness + spacing - thickness) / spacing
        out_num_slices = num_slices // np.ceil(step)
        if out_num_slices < target_slices//2:
            step = calc_slice_stepv2(
                thickness,
                spacing,
                exp_thickness//2,
                num_slices,
                target_slices=target_slices,
                default_step=step
            )
        return np.ceil(step)


# mip (maximum)
def calc_mip(imgs, slices_num=5):
    mip_img = []

    for i in range(0, imgs.shape[-1], slices_num):
        end = min(imgs.shape[-1], i + slices_num)
        mip_img.append(np.max(imgs[:, :, i:end], axis=2))
    mip_img = np.stack(mip_img, axis=2)  # (H, W, C)
    return mip_img


# minip (minimum)
def calc_minip(imgs, slices_num=5):
    minip_img = []

    for i in range(0, imgs.shape[-1], slices_num):
        end = min(imgs.shape[-1], i + slices_num)
        minip_img.append(np.min(imgs[:, :, i:end], axis=2))
    minip_img = np.stack(minip_img, axis=2)  # (H, W, C)
    return minip_img


# median
def calc_mid(imgs, slices_num=5):
    mid_img = []

    for i in range(0, imgs.shape[-1], slices_num):
        end = min(imgs.shape[-1], i + slices_num)
        mid_img.append(imgs[:, :, (i+end)//2])
    mid_img = np.stack(mid_img, axis=2)  # (H, W, C)
    return mid_img


def scale_depth(volume, max_depth, spacial=3, two_side: bool = True):
    if volume.shape[-1] > max_depth:
        diff = volume.shape[-1] - max_depth
        if two_side:
            volume = volume[..., int(diff//2) : -int(diff - diff//2)]
        else:
            volume = volume[..., :-int(diff)]
    elif volume.shape[-1] < max_depth:
        diff = int(max_depth - volume.shape[-1])
        if two_side:
            npad = (*[(0, 0)]*spacial, (diff//2, diff - diff//2))
        else:
            npad = (*[(0, 0)]*spacial, (0, diff))
        volume = np.pad(volume, pad_width=npad, mode='constant', constant_values=0)
    return volume


def get_mask_idx(mask, px_threshold=40, extend_num=1, continuous_id=False):
    r'''
    get the interested range
    mask: (h, w, d)
    '''
#     mask = np.transpose(mask, (2, 0, 1))  # (d, h, w)
    mask = np.where(mask > 0, 1, 0)

    msk_sum = mask.sum(axis=(0, 1))
    msk_sum = np.where(msk_sum > px_threshold, msk_sum, 0)
    msk_idx = np.where(msk_sum > 0)[0]
    if continuous_id:
        diff = np.diff(msk_idx)
        diff_idx = np.where(diff != 1)[0]
        if len(diff_idx) == 0:
            pass
        elif (len(diff_idx) == 1) & (diff_idx[0] < len(diff)//2):
            msk_idx = msk_idx[(diff_idx[0] + 1):]
        elif (len(diff_idx) == 1) & (diff_idx[0] >= len(diff)//2):
            msk_idx = msk_idx[:diff_idx[0]]
        elif len(diff_idx) > 2:
            idx_range = diff_idx[-1] - diff_idx[0]
            if idx_range < int(mask.shape[-1]*0.1):
                pass
            else:
                msk_idx = msk_idx[(diff_idx[0] + 1):diff_idx[-1]]
    start = np.max([msk_idx[0] - extend_num, 0])
    end = np.min([msk_idx[-1] + extend_num, mask.shape[-1] - 1])
    return start, end


def find_mask_margin(mask, class_id, axis=0, extend_num=1, default_start=None, default_end=None):
    r'''
    get the interested range
    mask: (h, w, d)
    '''
    mask = np.where(mask == class_id, 1, 0)  # (h, w, d)

    if axis == 0:
        sum_axis = (1, 2)
        total_num = mask.shape[0]
    elif axis == 1:
        sum_axis = (0, 2)
        total_num = mask.shape[1]
    elif axis == 2:
        sum_axis = (0, 1)
        total_num = mask.shape[2]

    if default_start is None:
        default_start = 0

    if default_end is None:
        default_end = total_num - 1

    msk_sum = mask.sum(axis=sum_axis)
    msk_idx = np.where(msk_sum > 0)[0]
    if len(msk_idx) == 0:
        start = 0
        end = total_num - 1
    else:
        start = np.max([msk_idx[0] - extend_num, default_start])
        end = np.min([msk_idx[-1] + extend_num, default_end])
    return start, end


def find_width(img, px_thres=49, thres_ratio=7, extend_ratio=0.05, vol_ratio=0.8):
    r'''
    img: h, w, d
    '''
    h, w, d = img.shape
    erode = int(np.ceil(d*(1 - vol_ratio))//2)
    img = img[:, :, erode:(-erode)]

    img = np.transpose(img, (2, 0, 1))  # (d, h, w)
    img = np.where(img > px_thres, 1, 0)
    dist = img.sum(axis=(0, 1))
    max_id = np.argmax(dist)
    thres = dist[max_id]/thres_ratio
    dist_bool = dist > thres

    for i in range(max_id, len(dist)):
        if not dist_bool[i]:
            w_max = i
            break
    else:
        w_max = len(dist)

    for i in range(max_id, 0, -1):
        if not dist_bool[i]:
            w_min = i
            break
    else:
        w_min = 0

    extend = np.ceil((w_max - w_min)*extend_ratio)
    w_min = max(0, int(w_min - extend // 2))
    w_max = min(img.shape[-1], int(w_max + extend // 2))
    if (w_max - w_min) < w//4:
        if w_min > w//2:
            w_min = 0
        elif w_min < w//2:
            w_max = w
    return w_min, w_max


def find_height(img, px_thres=33, thres_ratio=12, extend_ratio=0.1, vol_ratio=0.8):
    r'''
    img: h, w, d
    '''
    h, w, d = img.shape
    erode = int(np.ceil(d*(1 - vol_ratio))//2)
    img = img[:, :, erode:(-erode)]

    img = np.transpose(img, (2, 0, 1))  # (d, h, w)
    img = np.where(img > px_thres, 1, 0)
    dist = img.sum(axis=(0, 2))
    max_id = np.argmax(dist)
    thres = dist[max_id]/thres_ratio
    dist_bool = dist > thres

    for i in range(max_id, len(dist)):
        if not dist_bool[i]:
            h_max = i
            break
    else:
        h_max = len(dist)

    for i in range(max_id, 0, -1):
        if not dist_bool[i]:
            h_min = i
            break
    else:
        h_min = 0

    extend = np.ceil((h_max - h_min)*extend_ratio)
    h_min = max(0, int(h_min - extend // 2))
    h_max = min(img.shape[1], int(h_max + extend // 2))
    if (h_max - h_min) < h//4:
        if h_min > h//2:
            h_min = 0
        elif h_min < h//2:
            h_max = h
    return h_min, h_max


def resize_volume(volume, resize: tuple):
    r'''
    volume: (h, w, d)
    resize: (w, h)
    '''
    imgs = [cv2.resize(volume[:, :, i], resize, interpolation=cv2.INTER_AREA) for i in range(volume.shape[-1])]
    imgs = np.stack(imgs, axis=2)
    return imgs
