import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn


def custom_metric(labels: dict, output: tuple, logit: bool = True, infer: bool = False):
    output = list(output)
    if logit:
        output[0] = torch.sigmoid(output[0])
        output[1] = torch.sigmoid(output[1])
        output[2] = torch.softmax(output[2], dim=1)
        output[3] = torch.softmax(output[3], dim=1)
        output[4] = torch.softmax(output[4], dim=1)
        
    bowel_output = output[0]
    bowel_output = torch.concat([(1 - bowel_output), bowel_output], dim=1).detach().cpu().numpy()
    bowel_label = labels['bowel'].detach().cpu().numpy()
    bowel_weight = np.where(bowel_label == 1, 2, 1).reshape(-1)
    bowel_label = np.concatenate([(1 - bowel_label), bowel_label], axis=1)
    
    ext_output = output[1]
    ext_output = torch.concat([(1 - ext_output), ext_output], dim=1).detach().cpu().numpy()
    ext_label = labels['extravasation'].detach().cpu().numpy()
    ext_weight = np.where(ext_label == 1, 6, 1).reshape(-1)
    ext_label = np.concatenate([(1 - ext_label), ext_label], axis=1)
    
    kidney_output = output[2].detach().cpu().numpy()
    kidney_label = labels['kidney'].detach().cpu().numpy().astype(int)
    kidney_weight = np.where(kidney_label == 1, 2, np.where(kidney_label == 2, 4, 1)).squeeze()
#     kidney_label = np.eye(3)[kidney_label.astype(int)].squeeze()
    
    liver_output = output[3].detach().cpu().numpy()
    liver_label = labels['liver'].detach().cpu().numpy().astype(int)
    liver_weight = np.where(liver_label == 1, 2, np.where(liver_label == 2, 4, 1)).squeeze()
#     liver_label = np.eye(3)[liver_label.astype(int)].squeeze()
    
    spleen_output = output[4].detach().cpu().numpy()
    spleen_label = labels['spleen'].detach().cpu().numpy().astype(int)
    spleen_weight = np.where(spleen_label == 1, 2, np.where(spleen_label == 2, 4, 1)).squeeze()
#     spleen_label = np.eye(3)[spleen_label.astype(int)].squeeze()
    
    # any injury
    any_injury_label = labels['any_injury'].detach().cpu().numpy()
    any_injury_weight = np.where(any_injury_label == 1, 6, 6).squeeze()
    any_injury_output = np.stack([
        (1 - bowel_output[:, 0]),
        (1 - ext_output[:, 0]),
        (1 - kidney_output[:, 0]),
        (1 - liver_output[:, 0]),
        (1 - spleen_output[:, 0]),
    ], axis=1)
    any_injury_output = np.max(any_injury_output, axis=1, keepdims=True)
    
    bowel_loss = metrics.log_loss(bowel_label, bowel_output, sample_weight=bowel_weight, normalize=True)
    ext_loss = metrics.log_loss(ext_label, ext_output, sample_weight=ext_weight, normalize=True)
    # solid organ
    kidney_loss = metrics.log_loss(kidney_label, kidney_output, sample_weight=kidney_weight, labels=[0, 1, 2])
    liver_loss = metrics.log_loss(liver_label, liver_output, sample_weight=liver_weight, labels=[0, 1, 2])
    spleen_loss = metrics.log_loss(spleen_label, spleen_output, sample_weight=spleen_weight, labels=[0, 1, 2])
    # any injury -> max(1 - p(healthy))
    any_injury_loss = metrics.log_loss(any_injury_label, any_injury_output, sample_weight=any_injury_weight, labels=[0, 1])
    if infer:
        return bowel_loss, ext_loss, kidney_loss, liver_loss, spleen_loss, any_injury_loss
    else:
        return np.mean([bowel_loss, ext_loss, kidney_loss, liver_loss, spleen_loss, any_injury_loss])    
    

def any_injury_loss_fn(output, labels, weight=6):
    device = output[0].device
    output = list(output)
    any_injury_labels = labels['any_injury']
    any_injury_weight = torch.where(any_injury_labels == 1, weight, 1)
    bce_loss_fn = nn.BCELoss(weight=any_injury_weight, reduction='mean')
    
    output[0] = torch.sigmoid(output[0])
    output[1] = torch.sigmoid(output[1])
    output[2] = torch.softmax(output[2], dim=1)
    output[3] = torch.softmax(output[3], dim=1)
    output[4] = torch.softmax(output[4], dim=1)
    
    bowel_injury = 1 - output[0]
    ext_injury = 1 - output[1]
    kidney_injury = torch.softmax(output[2], dim=1)[:, 1:].sum(axis=1, keepdim=True)
    liver_injury = torch.softmax(output[3], dim=1)[:, 1:].sum(axis=1, keepdim=True)
    spleen_injury = torch.softmax(output[4], dim=1)[:, 1:].sum(axis=1, keepdim=True)
    
    injury = torch.stack([bowel_injury, ext_injury, kidney_injury, liver_injury, spleen_injury], axis=2)
    any_injury_prob = torch.max(injury, axis=2)[0]
    loss = bce_loss_fn(any_injury_prob, any_injury_labels)
    return loss


def custom_loss(output: tuple, labels: dict, any_injury=True, reduction='sum'):
    r'''
    bowel, extravasation, kidney, liver, spleen
    '''
    device = output[0].device
    # bowel weight:1:2
    bce_loss_fn_bowel = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([2]).to(device))
    # ext weight: 1:6
    bce_loss_fn_ext = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([6]).to(device))
    # solid organ: 1:2:4
    ce_loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.05, weight=torch.Tensor([1, 2, 4]).to(device))
    
    # binary
    bowel_loss = bce_loss_fn_bowel(output[0], labels['bowel'])
    ext_loss = bce_loss_fn_ext(output[1], labels['extravasation'])
    # multiclass
    kidney_loss = ce_loss_fn(output[2], labels['kidney'].squeeze().type(torch.LongTensor).to(device))
    liver_loss = ce_loss_fn(output[3], labels['liver'].squeeze().type(torch.LongTensor).to(device))
    spleen_loss = ce_loss_fn(output[4], labels['spleen'].squeeze().type(torch.LongTensor).to(device))
    # sum
    total_loss = bowel_loss + ext_loss + kidney_loss + liver_loss + spleen_loss
    mean_total_loss = total_loss / 5
    # any injury loss
    if any_injury:
        any_injury_loss = any_injury_loss_fn(output, labels)
        total_loss += any_injury_loss
        mean_total_loss = mean_total_loss / 6
    if reduction == 'sum':
        return total_loss
    else:
        return mean_total_loss


def custom_loss_v2(output: tuple, labels: dict, any_injury=True, reduction='sum'):
    r'''
    bowel, extravasation, kidney, liver, spleen
    '''
    device = output[0].device
    # bowel weight:1:2
    # bce_loss_fn_bowel = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([2]).to(device))
    bce_loss_fn_bowel = nn.BCEWithLogitsLoss(reduction='mean')
    # ext weight: 1:6
    # bce_loss_fn_ext = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([6]).to(device))
    bce_loss_fn_ext = nn.BCEWithLogitsLoss(reduction='mean')
    # solid organ: 1:2:4
    # ce_loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.05, weight=torch.Tensor([1, 2, 4]).to(device))
    ce_loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.05)
    
    # binary
    bowel_loss = bce_loss_fn_bowel(output[0], labels['bowel'])
    ext_loss = bce_loss_fn_ext(output[1], labels['extravasation'])
    # multiclass
    kidney_loss = ce_loss_fn(output[2], labels['kidney'].squeeze().type(torch.LongTensor).to(device))
    liver_loss = ce_loss_fn(output[3], labels['liver'].squeeze().type(torch.LongTensor).to(device))
    spleen_loss = ce_loss_fn(output[4], labels['spleen'].squeeze().type(torch.LongTensor).to(device))
    # sum
    total_loss = bowel_loss + ext_loss + kidney_loss + liver_loss + spleen_loss
    mean_total_loss = total_loss / 5
    # any injury loss
    if any_injury:
        any_injury_loss = any_injury_loss_fn(output, labels, weight=1)
        total_loss += any_injury_loss
        mean_total_loss = mean_total_loss / 6
    if reduction == 'sum':
        return total_loss
    else:
        return mean_total_loss

    


def custom_loss_organ(output: tuple, labels: dict, reduction='sum'):
    r'''
    solid organs: kidney, liver, spleen
    '''
    device = output[0].device
    ce_loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.05, weight=torch.Tensor([1, 2, 4]).to(device))
    
    # multiclass
    kidney_loss = ce_loss_fn(output[0], labels['kidney'].squeeze().to(torch.long))
    liver_loss = ce_loss_fn(output[1], labels['liver'].squeeze().to(torch.long))
    spleen_loss = ce_loss_fn(output[2], labels['spleen'].squeeze().to(torch.long))
    
    if reduction == 'sum':
        return kidney_loss + liver_loss + spleen_loss
    return (kidney_loss + liver_loss + spleen_loss)/3


def custom_metric_organ(labels: dict, output: tuple, logit: bool = True, infer=False):
    r'''
    solid organs: kidney, liver, spleen
    '''
    output = list(output)
    if logit:
        output[0] = torch.softmax(output[0], dim=1)
        output[1] = torch.softmax(output[1], dim=1)
        output[2] = torch.softmax(output[2], dim=1)
    
    kidney_output = output[0].detach().cpu().numpy()
    kidney_label = labels['kidney'].detach().cpu().numpy().astype(int)
    kidney_weight = np.where(kidney_label == 1, 2, np.where(kidney_label == 2, 4, 1)).squeeze()
    
    liver_output = output[1].detach().cpu().numpy()
    liver_label = labels['liver'].detach().cpu().numpy().astype(int)
    liver_weight = np.where(liver_label == 1, 2, np.where(liver_label == 2, 4, 1)).squeeze()
    
    spleen_output = output[2].detach().cpu().numpy()
    spleen_label = labels['spleen'].detach().cpu().numpy().astype(int)
    spleen_weight = np.where(spleen_label == 1, 2, np.where(spleen_label == 2, 4, 1)).squeeze()
    
    # solid organ
    kidney_loss = metrics.log_loss(kidney_label, kidney_output, sample_weight=kidney_weight, labels=[0, 1, 2])
    liver_loss = metrics.log_loss(liver_label, liver_output, sample_weight=liver_weight, labels=[0, 1, 2])
    spleen_loss = metrics.log_loss(spleen_label, spleen_output, sample_weight=spleen_weight, labels=[0, 1, 2])
    if infer:
        return kidney_loss, liver_loss, spleen_loss
    return np.mean([kidney_loss, liver_loss, spleen_loss])
