import torch
import segmentation_models_pytorch as smp
from rsna_model_utils import convert_2dto3d


# utility function
def predict_mask(x: np.ndarray, logit: bool = True, labelmap: bool = False):
    '''
    Convert ouputs into mask (0, 1).
    Input should be `numpy` format.
    '''
    if logit:
        x = torch.tensor(x)
        x = torch.sigmoid(x)
        x = x.numpy()

    if x.ndim not in [3, 4, 5]:
        raise ValueError(f"expect input dimention is 3 or 4, got {x.ndim}")

    if x.ndim == 4:
        c, *_ = x.shape
        pred = np.argmax(x, axis=0)  # shape: (height, width)
    else:
        b, c, *_ = x.shape
        pred = np.argmax(x, axis=1)  # shape: (batch, height, width)

    if not labelmap:
        pred = np.eye(c)[pred]
        pred = np.transpose(pred, (0, 4, 1, 2, 3))
    return pred


r'''
Model input: (B, C, D, H, W) -> (B, 1, 64, 256, 256)
    - value range: 0 to 1
    - dtype: torch.FloatTensor
    - windows: {'wl': 50, 'ww': 350}
Model output: (B, 5, D, H, W) -> (B, 5, 64, 256, 256)

mask classes:
    {'background': 0, 'liver': 1, 'spleen': 2, 'left_kidney': 3, 'right_kidney': 3, 'bowel': 4}
'''
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=5,
)
model = convert_2dto3d(model)

model_weight_path = "/home/doriskao/project/rsna_abdomen/models/seg_resnet34_best_loss_resize.pth"
model_weight = torch.load(model_weight_path, map_location='cpu')
model.load_state_dict(model_weight)


# inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TODO: read input data
images = torch.from_numpy(images).to(device, dtype=torch.float)
output = model(images)  # (b, c, d, h, w)

output = output.detach().cpu().numpy()
output_mask = predict_mask(output, labelmap=True)  # (b, d, h, w)
output_mask = np.transpose(output_mask.squeeze(), (1, 2, 0))  # (d, h, w) -> (h, w, d)
