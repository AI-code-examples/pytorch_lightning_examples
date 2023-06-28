#!/usr/bin/env python
#  -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
-----------------------------------------------
@Software   :   PyCharm
@Project    :   pytorch_lightning_examples
@File       :   image_net_resnet50
@Version    :   v0.1
@Time       :   2023/6/28 16:12
@License    :   (C)Copyright    2018-2023,  zYx.Tom
@Reference  :   https://lightning.ai/docs/pytorch/latest/advanced/transfer_learning.html
@Description:   Imagenet(Computer Vision)
@Thought    :
"""
import os
from typing import Any

import lightning.pytorch as pl
import torch
import torchvision.models as models
from lightning import Trainer
from torch import nn


class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # 初始化预训练模型 resnet
        backbone = models.resnet50(weights='DEFAULT')
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # 使用预训练模型分类 cifar-10（10个图像类别）
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)
        pass

    def forward(self, x) -> Any:
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
            pass
        x = self.classifier(representations)
        return x


def main():
    # 精调模型
    model = ImagenetTransferLearning()
    trainer = Trainer()
    trainer.fit(model)
    pass


def load_module(x):
    model = ImagenetTransferLearning.load_from_checkpoint(checkpoint_path)
    model.freeze()
    predictions = model(x)
    return predictions


# ----------------------------------------------------------------------
# 小结
if __name__ == '__main__':
    default_root_dir = os.path.join(os.getcwd(), 'outputs')
    checkpoint_path = os.path.join(default_root_dir, 'checkpoints/checkpoint.ckpt')
    main()
