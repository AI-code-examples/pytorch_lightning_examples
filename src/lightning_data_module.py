#!/usr/bin/env python
#  -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
-----------------------------------------------
@Software   :   PyCharm
@Project    :   pytorch_lightning_examples
@File       :   lightning_data_module
@Version    :   v0.1
@Time       :   2023/6/29 10:00
@License    :   (C)Copyright    2018-2023,  zYx.Tom
@Reference  :   
@Description:   
@Thought    :
"""
import os
from typing import Any
from typing import Dict

import lightning.pytorch as pl
import torchvision.transforms as transforms
from lightning import Trainer
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST

from lightning_basic_module import LitAutoEncoder


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'data', batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.current_train_batch_index = None
        pass

    def prepare_data(self) -> None:
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
            pass
        if stage == 'test':
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            pass
        if stage == 'predict':
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # 运行结束后，清理现场
        pass

    def state_dict(self) -> Dict[str, Any]:
        # 将 DataModule 的状态保存到 checkpoint
        state = {'current_train_batch_index': self.current_train_batch_index}
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # 将 DataModule 的状态从 checkpoint 中恢复
        self.current_train_batch_index = state_dict['current_train_batch_index']


def main():
    home_path = os.getcwd()
    data_path = os.path.join(home_path, 'data')
    mnist = MNISTDataModule(data_path)
    model = LitAutoEncoder()
    trainer = Trainer()
    simple_call(mnist, model, trainer)
    custom_call(mnist, model, trainer)
    pass


def custom_call(mnist, model, trainer):
    mnist.prepare_data()
    mnist.setup(stage='fit')
    trainer.fit(model=model, train_dataloaders=mnist)
    trainer.validate(dataloaders=mnist)
    mnist.setup(stage='test')
    trainer.test(dataloaders=mnist)


def simple_call(mnist, model, trainer):
    trainer.fit(model, datamodule=mnist)
    trainer.test(datamodule=mnist)
    trainer.validate(datamodule=mnist)
    trainer.predict(datamodule=mnist)


# ----------------------------------------------------------------------
# 小结
if __name__ == '__main__':
    main()
