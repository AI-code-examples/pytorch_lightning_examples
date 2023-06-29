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
import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.utilities.model_summary import ModelSummary
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
        self.example_input_array = torch.Tensor(32, 1, 28, 28)  # 模型每层输出的维度
        pass

    def forward(self, x) -> Any:
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, x

    def batch_step(self, batch):
        x, _ = batch
        x_hat, x = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        train_loss = self.batch_step(batch)
        self.any_lightning_module_function_or_hook()
        return train_loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        val_loss = self.batch_step(batch)
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        test_loss = self.batch_step(batch)
        self.log('test_loss', test_loss)
        return test_loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # ToDo: 也可以支持复杂的前处理或者后处理逻辑
        return self.batch_step(batch)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def any_lightning_module_function_or_hook(self):
        tensorboard_logger = self.logger.experiment
        fake_images = torch.Tensor(torch.randn((2, 1, 28, 28)))
        tensorboard_logger.add_images('generated_images', fake_images, 0)
        # tensorboard_logger.log_graph(model=self, input_array=fake_images)


def main():
    train_loader, valid_loader, test_loader, predict_loader = prepare_dataloader()
    # 初始化 lightning module
    auto_encoder = LitAutoEncoder(Encoder(), Decoder())
    # 不执行训练，输出模型结构
    summary = ModelSummary(auto_encoder, max_depth=-1)
    print(summary)
    lightning_auto_train(auto_encoder, train_loader, valid_loader, test_loader, predict_loader)
    export_onnx(auto_encoder)

    # lightning_manual_train(auto_encoder, train_loader)
    pass


def inference_onnx():
    """基于 onnx 执行推理

    :return:
    """
    ort_session = onnxruntime.InferenceSession(deploy_file_path)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: np.random.randn(32, 1, 28, 28).astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)
    pass


def export_onnx(auto_encoder):
    """ 输出模型为 onnx 格式

    :param auto_encoder:
    :return:
    """
    # 设置了 example_input_array，可以不设置 input_sample，直接输出模型
    auto_encoder.to_onnx(deploy_file_path, export_params=True)
    # input_sample = torch.randn((1, 1, 28, 28))
    # auto_encoder.to_onnx(deploy_path, input_sample, export_params=True)


def prepare_dataloader():
    # 数据集准备
    transform = transforms.ToTensor()  # 数据变换操作列表
    data_path = os.path.join(home_path, 'data')
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
    # 预测集
    predict_set = MNIST(root=data_path, download=True, train=False, transform=transform)
    predict_loader = DataLoader(predict_set, num_workers=2)
    return train_loader, valid_loader, test_loader, predict_loader


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


def lightning_auto_train(auto_encoder, train_loader, valid_loader, test_loader, predict_loader):
    """自动使用GPU计算

    :param auto_encoder:
    :param train_loader:
    :param valid_loader:
    :param test_loader:
    :param predict_loader:
    :return:
    """
    # 早停法的定义与设置
    early_stop_callbacks = [
        EarlyStopping(monitor='val_loss', mode='min'),
        EarlyStopping(monitor='val_accuracy', min_delta=0.00, patience=3, verbose=False, mode='max')
    ]
    profiler = AdvancedProfiler(dirpath=output_path, filename='perf_logs')  # 性能分析详细到函数调用
    logger = TensorBoardLogger(save_dir=output_path, log_graph=True, default_hp_metric=None)
    logger._log_graph = True
    logger._default_hp_metric = None
    trainer = pl.Trainer(
        # limit_train_batches=100,
        # limit_val_batches=10,
        # limit_test_batches=10,  # 只执行 100 条训练数据
        limit_train_batches=0.01,
        limit_val_batches=0.01,
        limit_test_batches=0.01,
        limit_predict_batches=0.01,  # 只执行 1% 的数据
        # fast_dev_run=True,  # 只执行一次，不执行验证集与测试集
        # fast_dev_run=3,  # 只执行三次，不执行验证集与测试集
        # num_sanity_val_steps=2,  # 做两次验证集检测，保证结果是靠谱的
        # callbacks=early_stop_callbacks,   # 模型训练过程中回调函数
        enable_checkpointing=False,  # 关闭模型输出
        enable_model_summary=False,  # 关闭模型结构输出
        # profiler='simple',  # 最简单的输出内容
        # profiler='pytorch',  # 只输出与 pytorch 相关的内容
        # profiler=profiler,  # 以文件形式输出性能分析结果
        logger=logger,
        max_epochs=1,
        default_root_dir=output_path)  # 日志与权重的输出路径
    trainer.fit(auto_encoder, train_loader, valid_loader)  # 模型训练与验证
    trainer.test(auto_encoder, test_loader)  # 模型测试
    trainer.predict(auto_encoder, predict_loader)


def load_module(x):
    model = LitAutoEncoder.load_from_checkpoint(checkpoint_path)
    model.eval()
    y_hat = model(x)
    return y_hat


class CIFAR10Classifier(pl.LightningModule):
    """使用LitAutoEncoder作为预训练模型

    """

    def __init__(self):
        super().__init__()
        self.feature_extractor = LitAutoEncoder.load_from_checkpoint(checkpoint_path)
        self.feature_extractor.freeze()
        self.classifier = nn.Linear(100, 10)
        pass

    def forward(self, x) -> Any:
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        return x


def predict():
    model = LitAutoEncoder.load_from_checkpoint(checkpoint_path)
    model.eval()
    x = torch.randn(1, 64)
    with torch.no_grad():
        y_hat = model(x)
        pass
    return y_hat


# ----------------------------------------------------------------------
# 小结
if __name__ == '__main__':
    home_path = os.getcwd()
    output_path = os.path.join(home_path, 'outputs')
    checkpoint_path = os.path.join(output_path, 'checkpoints/checkpoint.ckpt')
    deploy_file_path = os.path.join(output_path, 'deploy', 'onnx', 'LitAutoEncoder.onnx')

    main()
    # predict()
    # inference_onnx()
