# *_*coding: utf-8 *_*
# author --Lee--

import os
import torch
import torch.nn as nn
from torchvision import models

from opt import parse_opt
from data_load import train_val_test_data_process

from utils.LRAcc import LRAccuracyScheduler
from utils.Trainer import Trainer
from utils.EarlyStop import EarlyStopping

# 配置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_loader, val_loader, test_loader = train_val_test_data_process()


class ModifiedEfficientB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Use the EfficientNet B0 features part
        self.features = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).features
        # Adaptive average pooling layer
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out


if __name__ == '__main__':
    opt = parse_opt()

    model = ModifiedEfficientB0(num_classes=opt.num_classes)
    model_name = os.path.join("densenet121", str(opt.attention), str(opt.re_train), str(opt.monitor))

    # 损失
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opt.lr / 10,
                                  weight_decay=opt.weight_decay)
    lr_scheduler = LRAccuracyScheduler(optimizer, mode='max', patience=5, factor=0.1)
    early_stopping = EarlyStopping(patience=opt.patience, delta=0, monitor=opt.monitor)
    model_path = os.path.join(str(opt.save_model), model_name)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    trainer = Trainer(model, train_loader, val_loader, test_loader, criterion, optimizer, lr_scheduler, early_stopping)
    trainer.train(opt.epochs, model_path)

    initial_checkpoint = model_path + "/" + os.listdir(model_path)[0]
    test_loss, test_acc = trainer.test(test_loader, True, opt.dataset + '/test', initial_checkpoint)
    print('test_acc = ', test_acc)
    print('test_loss = ', test_loss)
