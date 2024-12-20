import warnings

import torch
from tqdm import tqdm

from utils import *

warnings.filterwarnings('ignore')

MODEL_PATH = ''


def model_predict(model, test_loader, adata=None, attention=True, device=torch.device('cpu')):
    model.eval()
    model = model.to(device)
    preds = None

    with torch.no_grad():
        for patch, position, exp, center, adj in tqdm(test_loader):
            patch, position, adj = patch.to(device), position.to(device), adj.to(device)
            pred = model(patch, position, adj)

            if preds is None:
                preds = pred.squeeze()
                ct = center
                gt = exp
            else:
                preds = torch.cat((preds, pred), dim=0)
                ct = torch.cat((ct, center), dim=0)
                gt = torch.cat((gt, exp), dim=0)  #
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()

    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct

    return adata, adata_gt


def sr_predict(model, test_loader, device=torch.device('cpu')):
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, center in tqdm(test_loader):

            patch, position = patch.to(device), position.to(device)
            pred = model(patch, position)

            if preds is None:
                preds = pred.squeeze()
                ct = center
            else:
                preds = torch.cat((preds, pred), dim=0)
                ct = torch.cat((ct, center), dim=0)
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    return adata
