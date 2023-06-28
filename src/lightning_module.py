#!/usr/bin/env python
#  -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
-----------------------------------------------
@Software   :   PyCharm
@Project    :   pytorch_lightning_examples
@File       :   lightning_module
@Version    :   v0.1
@Time       :   2023/6/28 9:09
@License    :   (C)Copyright    2018-2023,  zYx.Tom
@Reference  :   https://lightning.ai/docs/pytorch/latest/starter/introduction.html
@Description:   
@Thought    :
"""
import os
from typing import Any

import lightning.pytorch as pl
from torch import nn
from torch import optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


# 定义 LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    encoder = nn.Sequential(
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 3))
    decoder = nn.Sequential(
        nn.Linear(3, 64),
        nn.ReLU(),
        nn.Linear(64, 28 * 28)
    )
    auto_encoder = LitAutoEncoder(encoder, decoder)
    # 定义数据集，root为数据存储路径
    dataset = MNIST(root=os.path.join(os.getcwd(), 'data'), download=True, transform=ToTensor())
    train_loader = DataLoader(dataset)
    # 训练模型
    trainer = pl.Trainer(
        limit_train_batches=100,
        max_epochs=1,
        default_root_dir=os.path.join(os.getcwd(), 'outputs'))  # 日志与权重的输出路径
    trainer.fit(model=auto_encoder, train_dataloaders=train_loader)
    pass


# ----------------------------------------------------------------------
# 小结
if __name__ == '__main__':
    main()
