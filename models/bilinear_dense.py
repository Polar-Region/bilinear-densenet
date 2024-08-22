from torchvision import models

from models.attention_block import AttentionBlock, CBAM
from model.attention.A2Atttention import *
from model.attention.ACmixAttention import *
from model.attention.AFT import *
from model.attention.Axial_attention import *
from model.attention.BAM import *
from model.attention.CoAtNet import *
from model.attention.CoTAttention import *
from model.attention.DANet import *
from model.attention.ECAAttention import *
from model.attention.EMSA import *
from model.attention.ExternalAttention import *
from model.attention.HaloAttention import *
from model.attention.MobileViTAttention import *
from model.attention.MobileViTv2Attention import *
from model.attention.MUSEAttention import *
from model.attention.OutlookAttention import *
from model.attention.ParNetAttention import *
from model.attention.PolarizedSelfAttention import *
from model.attention.PSA import *
from model.attention.ResidualAttention import *
from model.attention.S2Attention import *


class BilinearDense121(nn.Module):
    def __init__(self, num_classes, attention="None", fc="fc"):
        super().__init__()
        # 使用DenseNet121的features部分作为卷积层
        self.conv = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if fc else None).features
        # 定义注意力机制层
        attentionCase = {
            "CBAM": CBAM(1024),
            "AttentionBlock": AttentionBlock(1024),
            "None": None,
            "A2Attention": DoubleAttention(1024, 128, 128),
            "ACmixAttention": ACmix(in_planes=1024, out_planes=1024),
            "AFT": AFT_FULL(d_model=1024, n=49),
            "Axial_attention": AxialImageTransformer(
                dim=1024,
                depth=7,
                reversible=True
            ),
            "BAM": BAMBlock(channel=1024, reduction=7, dia_val=2),
            "CoAtNet": CoAtNet(1024, 1024),
            "CoTAttention": CoTAttention(dim=1024, kernel_size=3),
            "DANeT": DAModule(d_model=512, kernel_size=3, H=7, W=7),
            "ECAAttention": ECAAttention(3),
            "EMSA": EMSA(d_model=512, d_k=512, d_v=512, h=8, H=8, W=8, ratio=2, apply_transform=True),
            "ExternalAttention": ExternalAttention(1024, 8),
            "HaloAttention": HaloAttention(dim=1024,
                                           block_size=2,
                                           halo_size=1, ),
            "MobileViTAttention": MobileViTAttention(dim=1024),
            "MobileViTv2Attention": MobileViTv2Attention(d_model=1024),
            "MUSEAttention": MUSEAttention(d_model=1024, d_k=1024, d_v=1024, h=8),
            "OutlookAttention": OutlookAttention(dim=1024),
            "ParNetAttention": ParNetAttention(1024),
            "PolaraizedSelfAttention": SequentialPolarizedSelfAttention(channel=1024),
            "PSA": PSA(channel=1024, reduction=8),
            "ResidualAttention": ResidualAttention(channel=1024, la=0.2),
            "S2Attention": S2Attention(1024)

        }
        self.attention = attentionCase[attention]
        # 定义全连接层，输入维度为1024*1024，输出维度为num_classes
        self.fc = nn.Linear(1024 * 1024, num_classes)
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
        # 使用注意力机制
        if self.attention:
            x = self.attention(x)
        x = torch.flatten(x, 2)
        # 进行交叉乘积操作，得到二维特征张量
        x = x.view(x.size(0), 1024, 7 * 7)
        x_t = torch.transpose(x, 1, 2)
        x = torch.bmm(x, x_t) / (7 * 7)
        x = x.view(x.size(0), 1024 * 1024)
        # 进行 The signed square root 操作
        features = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        # 进行 L2 归一化
        features = torch.nn.functional.normalize(features)

        out = self.fc(features)
        return out


class BilinearDense201(nn.Module):
    def __init__(self, num_classes, attention="None", fc="fc"):
        super().__init__()
        # 使用DenseNet121的features部分作为卷积层
        self.conv = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT if fc else None).features
        # 定义注意力机制层
        attentionCase = {
            "CBAM": CBAM(1920),
            "Attention": AttentionBlock(1920),
            "None": None,
            "A2Attention": DoubleAttention(1920, 128, 128),
            "ACmixAttention": ACmix(in_planes=1920, out_planes=1920),
            "AFT": AFT_FULL(d_model=1920, n=49),
            "Axial_attention": AxialImageTransformer(
                dim=1920,
                depth=7,
                reversible=True
            ),
            "BAM": BAMBlock(channel=1920, reduction=7, dia_val=2),
            "CoAtNet": CoAtNet(1920, 1920),
            "CoTAttention": CoTAttention(dim=1920, kernel_size=3),
            "DANeT": DAModule(d_model=512, kernel_size=3, H=7, W=7),
            "ECAAttention": ECAAttention(3),
            "EMSA": EMSA(d_model=512, d_k=512, d_v=512, h=8, H=8, W=8, ratio=2, apply_transform=True),
            "ExternalAttention": ExternalAttention(1920, 8),
            "HaloAttention": HaloAttention(dim=1920,
                                           block_size=2,
                                           halo_size=1, ),
            "MobileViTAttention": MobileViTAttention(dim=1920),
            "MobileViTv2Attention": MobileViTv2Attention(d_model=1920),
            "MUSEAttention": MUSEAttention(d_model=1920, d_k=1920, d_v=1920, h=8),
            "OutlookAttention": OutlookAttention(dim=1920),
            "ParNetAttention": ParNetAttention(1920),
            "PolaraizedSelfAttention": SequentialPolarizedSelfAttention(channel=1920),
            "PSA": PSA(channel=1920, reduction=8),
            "ResidualAttention": ResidualAttention(channel=1920, la=0.2),
            "S2Attention": S2Attention(1920)
        }
        self.attention = attentionCase[attention]
        # 定义全连接层，输入维度为1920*1920，输出维度为num_classes
        self.fc = nn.Linear(1920 * 1920, num_classes)
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
        # 使用注意力机制
        if self.attention:
            x = self.attention(x)
        x = torch.flatten(x, 2)
        # 进行交叉乘积操作，得到二维特征张量
        x = x.view(x.size(0), 1920, 7 * 7)
        x_t = torch.transpose(x, 1, 2)
        x = torch.bmm(x, x_t) / (7 * 7)
        x = x.view(x.size(0), 1920 * 1920)
        # 进行 The signed square root 操作
        features = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        # 进行 L2 归一化
        features = torch.nn.functional.normalize(features)

        out = self.fc(features)
        return out
