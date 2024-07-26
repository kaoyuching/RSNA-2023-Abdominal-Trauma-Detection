import numpy as np
import torch
from tqdm.auto import tqdm
from rsna_utils import custom_metric, custom_metric_organ


class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0
        
    def update(self, val):
        self.total += val
        self.steps += 1
        
    def __call__(self):
        return self.total / self.steps
    
    
def run_train(dataloader, model, loss_fn, optimizer, scheduler, device):
    loss_avg = RunningAverage()
    
    model.train()
    for image, labels in tqdm(dataloader):
        image = image.to(torch.float32).to(device)
        labels = {k: v.to(torch.float32).to(device) for k, v in labels.items()}

        output = model(image)
        loss = loss_fn(output, labels)
        loss_avg.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return loss_avg()


@torch.no_grad()
def run_valid(dataloader, model, loss_fn, device, true_labels: dict):
    loss_avg = RunningAverage()
    
    bowel_preds = []
    ext_preds = []
    kidney_preds = []
    liver_preds = []
    spleen_preds = []
    
    model.eval()
    for image, labels in tqdm(dataloader):
        image = image.type(torch.FloatTensor).to(device)
        labels = {k: v.type(torch.FloatTensor).to(device) for k, v in labels.items()}

        output = model(image)
        loss = loss_fn(output, labels)
        loss_avg.update(loss.item())
        
        bowel_preds.extend(list(output[0].detach().cpu().numpy()))
        ext_preds.extend(list(output[1].detach().cpu().numpy()))
        kidney_preds.extend(list(output[2].detach().cpu().numpy()))
        liver_preds.extend(list(output[3].detach().cpu().numpy()))
        spleen_preds.extend(list(output[4].detach().cpu().numpy()))

        # scores = custom_metric(labels, output, infer=True)
        
    preds = [
        torch.tensor([x for x in bowel_preds]),
        torch.tensor([x for x in ext_preds]),
        torch.tensor([x for x in kidney_preds]),
        torch.tensor([x for x in liver_preds]),
        torch.tensor([x for x in spleen_preds]),
    ]
    scores = custom_metric(true_labels, preds, infer=True)
    return loss_avg(), np.mean(scores), scores


@torch.no_grad()
def run_valid_organ(dataloader, model, loss_fn, device, true_labels: dict):
    loss_avg = RunningAverage()
    
    kidney_preds = []
    liver_preds = []
    spleen_preds = []
    
    model.eval()
    for image, labels in tqdm(dataloader):
        image = image.type(torch.FloatTensor).to(device)
        labels = {k: v.type(torch.FloatTensor).to(device) for k, v in labels.items()}

        output = model(image)
        loss = loss_fn(output, labels)
        loss_avg.update(loss.item())
        
        kidney_preds.extend(list(output[0].detach().cpu().numpy()))
        liver_preds.extend(list(output[1].detach().cpu().numpy()))
        spleen_preds.extend(list(output[2].detach().cpu().numpy()))

        # scores = custom_metric_organ(labels, output, logit=True, infer=True)
        
    preds = [
        torch.tensor([x for x in kidney_preds]),
        torch.tensor([x for x in liver_preds]),
        torch.tensor([x for x in spleen_preds]),
    ]
    scores = custom_metric_organ(true_labels, preds, infer=True)
    return loss_avg(), np.mean(scores), scores


def run_train_v2(dataloader, model, loss_fn, optimizer, scheduler, device):
    loss_avg = RunningAverage()
    
    model.train()
    for image, crop_kidney, crop_liver, crop_spleen, labels in tqdm(dataloader):
        image = image.to(torch.float32).to(device)
        crop_kidney = crop_kidney.to(torch.float32).to(device)
        crop_liver = crop_liver.to(torch.float32).to(device)
        crop_spleen = crop_spleen.to(torch.float32).to(device)
        labels = {k: v.to(torch.float32).to(device) for k, v in labels.items()}

        output = model(image, crop_kidney, crop_liver, crop_spleen)
        loss = loss_fn(output, labels)
        loss_avg.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return loss_avg()


@torch.no_grad()
def run_valid_v2(dataloader, model, loss_fn, device, true_labels: dict):
    loss_avg = RunningAverage()
    
    bowel_preds = []
    ext_preds = []
    kidney_preds = []
    liver_preds = []
    spleen_preds = []
    
    model.eval()
    for image, crop_kidney, crop_liver, crop_spleen, labels in tqdm(dataloader):
        image = image.type(torch.FloatTensor).to(device)
        crop_kidney = crop_kidney.type(torch.FloatTensor).to(device)
        crop_liver = crop_liver.type(torch.FloatTensor).to(device)
        crop_spleen = crop_spleen.type(torch.FloatTensor).to(device)
        labels = {k: v.type(torch.FloatTensor).to(device) for k, v in labels.items()}

        output = model(image, crop_kidney, crop_liver, crop_spleen)
        loss = loss_fn(output, labels)
        loss_avg.update(loss.item())
        
        bowel_preds.extend(list(output[0].detach().cpu().numpy()))
        ext_preds.extend(list(output[1].detach().cpu().numpy()))
        kidney_preds.extend(list(output[2].detach().cpu().numpy()))
        liver_preds.extend(list(output[3].detach().cpu().numpy()))
        spleen_preds.extend(list(output[4].detach().cpu().numpy()))

        # scores = custom_metric(labels, output, infer=True)
        
    preds = [
        torch.tensor([x for x in bowel_preds]),
        torch.tensor([x for x in ext_preds]),
        torch.tensor([x for x in kidney_preds]),
        torch.tensor([x for x in liver_preds]),
        torch.tensor([x for x in spleen_preds]),
    ]
    scores = custom_metric(true_labels, preds, infer=True)
    return loss_avg(), np.mean(scores), scores
