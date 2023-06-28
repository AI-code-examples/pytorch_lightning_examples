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
import torchvision.transforms as transforms
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
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
        train_loss = F.mse_loss(x_hat, x)
        return train_loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        valid_loss = F.mse_loss(x_hat, x)
        self.log('valid_loss', valid_loss)
        return valid_loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    train_loader, valid_loader, test_loader = prepare_dataloader()
    # 初始化 lightning module
    auto_encoder = LitAutoEncoder(Encoder(), Decoder())
    lightning_auto_train(auto_encoder, train_loader, valid_loader, test_loader)

    # lightning_manual_train(auto_encoder, train_loader)
    pass


def prepare_dataloader():
    # 数据集准备
    transform = transforms.ToTensor()  # 数据变换操作列表
    data_path = os.path.join(os.getcwd(), 'data')
    data_set = MNIST(root=data_path, download=True, train=True, transform=transform)
    data_set_size = len(data_set)
    train_set_size = int(data_set_size * 0.8)
    valid_set_size = data_set_size - train_set_size
    seed = torch.Generator().manual_seed(42)
    # 训练集,验证集
    train_set, valid_set = random_split(
        dataset=data_set,
        lengths=[train_set_size, valid_set_size],
        generator=seed)
    train_loader = DataLoader(train_set, num_workers=2)
    valid_loader = DataLoader(valid_set, num_workers=2)
    # 测试集
    test_set = MNIST(root=data_path, download=True, train=False, transform=transform)
    test_loader = DataLoader(test_set, num_workers=2)
    return train_loader, valid_loader, test_loader


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


def lightning_auto_train(auto_encoder, train_loader, valid_loader, test_loader):
    """自动使用GPU计算

    :param auto_encoder:
    :param train_loader:
    :param valid_loader:
    :param test_loader:
    :return:
    """
    trainer = pl.Trainer(
        max_epochs=1,
        default_root_dir=default_root_dir)  # 日志与权重的输出路径
    trainer.fit(auto_encoder, train_loader, valid_loader)
    trainer.test(auto_encoder, test_loader)


def load_module(x):
    model = LitAutoEncoder.load_from_checkpoint(os.path.join(default_root_dir, 'checkpoints/checkpoint.ckpt'))
    model.eval()
    y_hat = model(x)
    return y_hat


# ----------------------------------------------------------------------
# 小结
if __name__ == '__main__':
    default_root_dir = os.path.join(os.getcwd(), 'outputs')
    main()
