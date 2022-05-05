from typing import List
from matplotlib.axis import XAxis
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

from .data import MFCCData
from ..sound import MFCC

class ResLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.linear = nn.Linear(channels, channels)
        self.relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.relu(self.linear(x)) + x)

class MLPClassifier(pl.LightningModule):
    """感知机分类器。"""

    def __init__(self, num_of_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 1024),
            nn.ReLU(True),            
            *[ResLayer(1024) for _ in range(10)],
            nn.Linear(1024, num_of_classes),
        )

    def forward(self, x) -> torch.Tensor:
        probs = self.model(x)
        return probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer

    def analyze(self, mfcc: MFCC, mfcc_data: MFCCData) -> List[str]:
        """分析语音数据，预测每帧的浊音类别。"""
        frames = mfcc.data.shape[1] // 5
        X = mfcc.data.T[: frames * 5, :].reshape(frames, -1)
        label_probs = self.forward(torch.from_numpy(X))
        label = torch.argmax(label_probs, dim=1)
        return [mfcc_data.index_to_key[int(index)] for index in label]
