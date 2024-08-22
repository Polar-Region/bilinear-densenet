# -*- coding: utf-8 -*-
# author --Lee--

import torch
import torch.nn as nn
from torchvision import models

from models.attention_block import AttentionBlock, CBAM


class BilinearMobileNetV2(nn.Module):
    def __init__(self, num_classes, attention=None, fc="fc"):
        super().__init__()
        # 使用DenseNet121的features部分作为卷积层
        self.conv = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if fc else None).features
        # 定义注意力机制层
        self.attention = CBAM(1024) if attention == "CBAM" else AttentionBlock(1024) if attention else None
        # 定义全连接层，输入维度为1280*1280，输出维度为num_classes
        self.fc = nn.Linear(1280 * 1280, num_classes)
        if fc:
            # 将卷积层参数的requires_grad属性设为False，即冻结其参数
            for parameter in self.conv.parameters():
                parameter.requires_grad = False
            # 对全连接层参数进行初始化
            nn.init.kaiming_normal_(self.fc.weight.data)
            nn.init.constant_(self.fc.bias, val=0)

    def forward(self, x):
        # 前向传播，先经过卷积层，再进行平均池化，最后将输出展平为二维张量
        x = self.conv(x)
        x = torch.flatten(x, 2)
        # 进行交叉乘积操作，得到二维特征张量
        x = x.view(x.size(0), 1280, 7 * 7)
        x_t = torch.transpose(x, 1, 2)
        x = torch.bmm(x, x_t) / (7 * 7)
        x = x.view(x.size(0), 1280 * 1280)
        # 进行 The signed square root 操作
        features = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        # 进行 L2 归一化
        features = torch.nn.functional.normalize(features)

        out = self.fc(features)
        return out
    