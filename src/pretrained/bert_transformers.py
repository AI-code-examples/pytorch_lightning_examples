#!/usr/bin/env python
#  -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
-----------------------------------------------
@Software   :   PyCharm
@Project    :   pytorch_lightning_examples
@File       :   bert_transformers
@Version    :   v0.1
@Time       :   2023/6/28 16:21
@License    :   (C)Copyright    2018-2023,  zYx.Tom
@Reference  :
https://lightning.ai/docs/pytorch/latest/advanced/transfer_learning.html
https://github.com/huggingface/transformers
@Description:   BERT(NLP), Huggingface transformers
@Thought    :
"""
from typing import Any

import lightning.pytorch as pl


class BertMNLIFinetuner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased', output_attention=True)
        self.W = nn.Linear(self.bert.config.hidden_size, 3)
        self.num_classes = 3
        pass

    def forward(self, input_ids, attention_mask, token_type_ids) -> Any:
        h, _, attn = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn


def main():
    pass


# ----------------------------------------------------------------------
# 小结
if __name__ == '__main__':
    main()
