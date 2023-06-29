#!/usr/bin/env python
#  -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
-----------------------------------------------
@Software   :   PyCharm
@Project    :   pytorch_lightning_examples
@File       :   pretrained_modules
@Version    :   v0.1
@Time       :   2023/6/29 15:14
@License    :   (C)Copyright    2018-2023,  zYx.Tom
@Reference  :   
@Description:   
@Thought    :
"""
import os

from config import CHECKPOINT_PATH


def main():
    # Github URL where saved models are stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial5/"
    # Files to download
    pretrained_files = [
        "GoogleNet.ckpt",
        "ResNet.ckpt",
        "ResNetPreAct.ckpt",
        "DenseNet.ckpt",
        "tensorboards/GoogleNet/events.out.tfevents.googlenet",
        "tensorboards/ResNet/events.out.tfevents.resnet",
        "tensorboards/ResNetPreAct/events.out.tfevents.resnetpreact",
        "tensorboards/DenseNet/events.out.tfevents.densenet",
    ]
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if "/" in file_name:
            os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"请下载  {file_url}...")
    pass


# ----------------------------------------------------------------------
# 小结
if __name__ == '__main__':
    main()
