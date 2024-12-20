from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchmetrics.functional import accuracy

from GATLayer import MultiHeadGAT
from ODConv import ODConv2d
from efficient_capsnet import EfficientCapsNet
from transformer import ViT
from utils import *


class FeatureExtractor(nn.Module):
    """Some Information about FeatureExtractor"""

    def __init__(self, backbone='resnet101'):
        super(FeatureExtractor, self).__init__()
        backbone = torchvision.models.resnet101(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*layers)
        # self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        return x


class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=4, backbone='resnet50', learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        backbone = torchvision.models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_target_classes = num_classes
        self.classifier = nn.Linear(num_filters, num_target_classes)
        # self.valid_acc = torchmetrics.Accuracy()
        self.learning_rate = learning_rate

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.feature_extractor(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log('valid_loss', loss)
        self.log('valid_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.0001)
        return parser


class STModel(pl.LightningModule):
    def __init__(self, feature_model=None, n_genes=1000, hidden_dim=2048, learning_rate=1e-5, use_mask=False,
                 use_pos=False, cls=False):
        super().__init__()
        self.save_hyperparameters()
        # self.feature_model = None
        if feature_model:
            # self.feature_model = ImageClassifier.load_from_checkpoint(feature_model)
            # self.feature_model.freeze()
            self.feature_extractor = ImageClassifier.load_from_checkpoint(feature_model)
        else:
            self.feature_extractor = FeatureExtractor()
        # self.pos_embed = nn.Linear(2, hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, n_genes)

        self.learning_rate = learning_rate
        self.n_genes = n_genes

    def forward(self, patch, center):
        feature = self.feature_extractor(patch).flatten(1)
        h = feature
        pred = self.pred_head(F.relu(h))
        return pred

    def training_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        patch, center, exp, mask, label = batch
        if self.use_mask:
            pred, mask_pred = self(patch, center)
        else:
            pred = self(patch, center)

        loss = F.mse_loss(pred, exp)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


class THItoGene(pl.LightningModule):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.2, n_pos=64,
                 heads=[16, 8], caps=20, route_dim=64):
        super().__init__()

        self.learning_rate = learning_rate
        patch_dim = 3 * patch_size * patch_size
        self.route_dim = route_dim
        self.caps = caps

        self.relu = nn.ReLU()

        self.odconv2d = ODConv2d(in_planes=3, out_planes=16, kernel_size=4, stride=4)

        caps_out = (caps + 2) * route_dim

        self.caps_layer = EfficientCapsNet(rout_capsules=caps, route_dim=route_dim)

        self.x_embed = nn.Embedding(n_pos, route_dim)
        self.y_embed = nn.Embedding(n_pos, route_dim)

        self.vit = ViT(dim=caps_out, depth=n_layers, heads=heads[0], mlp_dim=2 * dim, dropout=dropout,
                       emb_dropout=dropout)

        self.gat = MultiHeadGAT(in_features=caps_out, nhid=1024, out_features=512, heads=heads[1], dropout=dropout,
                                alpha=0.01)

        self.gene_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, n_genes)
        )

    def forward(self, patches, centers, adj):
        B, N, C, H, W = patches.shape
        patches = patches.reshape(B * N, C, H, W)
        patches = self.odconv2d(patches)
        patches = self.relu(patches)

        patches = self.caps_layer(patches)
        patches = patches.reshape(-1, self.caps, self.route_dim)

        centers_x = self.x_embed(centers[:, :, 0]).permute(1, 0, 2)
        centers_y = self.y_embed(centers[:, :, 1]).permute(1, 0, 2)

        x = torch.concat((patches, centers_x, centers_y), dim=1)
        x = x.reshape(1, x.shape[0], -1)

        x = self.vit(x)
        x = x.reshape(x.shape[1], -1)

        x = self.gat(x, adj)
        x = self.gene_head(x)
        return x

    def training_step(self, batch, batch_idx):
        patch, center, exp, adj = batch
        pred = self(patch, center, adj)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp, adj = batch
        pred = self(patch, center, adj)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('valid_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        patch, center, exp, adj = batch
        pred = self(patch, center, adj)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('test_loss', loss, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        patch, position, exp, center, adj = batch
        pred = self(patch, position, adj)
        preds = pred.squeeze()
        ct = center
        gt = exp
        preds = preds.cpu().squeeze().numpy()
        ct = ct.cpu().squeeze().numpy()
        gt = gt.cpu().squeeze().numpy()
        adata = ann.AnnData(preds)
        adata.obsm['spatial'] = ct

        adata_gt = ann.AnnData(gt)
        adata_gt.obsm['spatial'] = ct

        return adata, adata_gt

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
