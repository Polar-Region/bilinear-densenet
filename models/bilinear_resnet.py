# -*- coding: utf-8 -*-
# author --Lee--

import torch
import torch.nn as nn
import torchvision.models as models

from models.attention_block import AttentionBlock, CBAM


class BilinearResNet50(nn.Module):
    def __init__(self, num_classes, attention=None, fc="fc"):
        super().__init__()
        # 使用ResNet50的features部分作为卷积层
        self.conv = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if fc else None)
        # 定义注意力机制层
        self.attention = CBAM(1024) if attention == "CBAM" else AttentionBlock(1024) if attention else None
        # 去掉ResNet50的全连接层
        self.conv = nn.Sequential(*list(self.conv.children())[:-1])
        # 定义全连接层，输入维度为2048，输出维度为num_classes
        self.fc = nn.Linear(2048 * 2048, num_classes)
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
        x = x.view(x.size(0), 2048, 1 * 1)
        x_t = torch.transpose(x, 1, 2)
        x = torch.bmm(x, x_t) / (1 * 1)
        x = x.view(x.size(0), 2048 * 2048)
        # 进行 The signed square root 操作
        features = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        # 进行 L2 归一化
        features = torch.nn.functional.normalize(features)

        out = self.fc(features)
        return out


class BilinearResNet101(nn.Module):
    def __init__(self, num_classes, attention=None, fc="fc"):
        super().__init__()
        # 使用ResNet50的features部分作为卷积层
        self.conv = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if fc else None)
        # 定义注意力机制层
        self.attention = CBAM(1024) if attention == "CBAM" else AttentionBlock(1024) if attention else None
        # 去掉ResNet101的全连接层
        self.conv = nn.Sequential(*list(self.conv.children())[:-1])
        # 定义全连接层，输入维度为2048，输出维度为num_classes
        self.fc = nn.Linear(2048 * 2048, num_classes)
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
        x = x.view(x.size(0), 2048, 1 * 1)
        x_t = torch.transpose(x, 1, 2)
        x = torch.bmm(x, x_t) / (1 * 1)
        x = x.view(x.size(0), 2048 * 2048)
        # 进行 The signed square root 操作
        features = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        # 进行 L2 归一化
        features = torch.nn.functional.normalize(features)

        out = self.fc(features)
        return out


class BilinearResNet152(nn.Module):
    def __init__(self, num_classes, attention=None, fc="fc"):
        super().__init__()
        # 使用ResNet50的features部分作为卷积层
        self.conv = models.resnet152(weights=models.ResNet152_Weights.DEFAULT if fc else None)
        # 定义注意力机制层
        self.attention = CBAM(1024) if attention == "CBAM" else AttentionBlock(1024) if attention else None
        # 去掉ResNet152的全连接层
        self.conv = nn.Sequential(*list(self.conv.children())[:-1])
        # 定义全连接层，输入维度为2048，输出维度为num_classes
        self.fc = nn.Linear(2048 * 2048, num_classes)
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
        x = x.view(x.size(0), 2048, 1 * 1)
        x_t = torch.transpose(x, 1, 2)
        x = torch.bmm(x, x_t) / (1 * 1)
        x = x.view(x.size(0), 2048 * 2048)
        # 进行 The signed square root 操作
        features = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        # 进行 L2 归一化
        features = torch.nn.functional.normalize(features)

        out = self.fc(features)
        return out