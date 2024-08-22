# -*- coding: utf-8 -*-
# author --Lee--

import torch
import torch.nn as nn

from opt import parse_opt
from data_load import train_val_test_data_process

from models.bilinear_dense import BilinearDense121, BilinearDense201


from utils.Trainer import Trainer
from utils.EarlyStop import EarlyStopping
from utils.LRAcc import LRAccuracyScheduler


import warnings

# 配置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_loader, val_loader, test_loader = train_val_test_data_process()

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    opt = parse_opt()
    if opt.net == "bd121_fc":
        model = BilinearDense121(opt.num_classes, opt.attention, None).to(device)
        model_name = "densenet121_all_" + opt.attention if opt.attention else "densenet121_all"

    elif opt.net == "bd201_fc":
        model = BilinearDense201(opt.num_classes, opt.attention, None).to(device)
        model_name = "densenet201_all_" + opt.attention if opt.attention else "densenet201_all"

    else:
        model = None
        model_name = None
        print("print error ,please choose the correct net again")
        exit()

    # 损失
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opt.lr,
                                  weight_decay=opt.weight_decay)
    lr_scheduler = LRAccuracyScheduler(optimizer, mode='max', patience=5, factor=0.1)
    early_stopping = EarlyStopping(patience=10, delta=0, monitor=opt.monitor)

    initial_checkpoint = \
        '/home/hipeson/Bilinear_Densenet/runs/save_model/Densenet201_all/best.pt'

    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, lr_scheduler, early_stopping)
    test_loss, test_acc = trainer.test(test_loader, 1, opt.dataset, initial_checkpoint)
    print('test_acc = ', test_acc)
    print('test_loss = ', test_loss)
