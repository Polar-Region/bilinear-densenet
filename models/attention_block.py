import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 计算特征图中每个位置的重要性权重
        weights = self.conv(x)
        weights = torch.sigmoid(weights)

        # 对特征图的每个切片进行加权
        x = x * weights

        # 返回加权后的特征图
        return x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # 空间注意力
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_out = self.sigmoid(avg_out + max_out)

        # 空间注意力
        spatial_out = self.conv1(x)
        spatial_out = self.bn1(spatial_out)
        spatial_out = self.sigmoid2(self.bn2(self.conv2(spatial_out)))
        spatial_out = self.conv3(spatial_out)

        # 组合通道和空间注意力
        x = x * channel_out * spatial_out

        return x
