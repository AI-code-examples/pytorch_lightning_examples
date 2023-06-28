#!/usr/bin/env python
#  -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
-----------------------------------------------
@Software   :   PyCharm
@Project    :   pytorch_lightning_examples
@File       :   lightning_basic_module
@Version    :   v0.1
@Time       :   2023/6/28 10:56
@License    :   (C)Copyright    2018-2023,  zYx.Tom
@Reference  :   
@Description:   
@Thought    :
"""
import os
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch.optim
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        pass

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28)
        )
        pass

    def forward(self, x):
        return self.l1(x)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        pass

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    dataset = MNIST(os.path.join(os.getcwd(), 'data'), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset)
    auto_encoder = LitAutoEncoder(Encoder(), Decoder())
    lightning_auto_train(auto_encoder, train_loader)

    # lightning_manual_train(auto_encoder, train_loader)
    pass


def lightning_manual_train(auto_encoder, train_loader):
    """默认使用CPU计算，GPU调度需要自己完成

    :param auto_encoder:
    :param train_loader:
    :return:
    """
    optimizer = auto_encoder.configure_optimizers()
    for batch_idx, batch in enumerate(train_loader):
        loss = auto_encoder.training_step(batch, batch_idx)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def lightning_auto_train(auto_encoder, train_loader):
    """自动使用GPU计算

    :param auto_encoder:
    :param train_loader:
    :return:
    """
    trainer = pl.Trainer(default_root_dir=os.path.join(os.getcwd(), 'outputs'))  # 日志与权重的输出路径
    trainer.fit(model=auto_encoder, train_dataloaders=train_loader)


# ----------------------------------------------------------------------
# 小结
if __name__ == '__main__':
    main()
