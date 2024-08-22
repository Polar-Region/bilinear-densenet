# *_*coding: utf-8 *_*
# author --Lee--

import os
import torch
import torch.nn as nn

from opt import parse_opt
from data_load import train_val_test_data_process

from utils.LRAcc import LRAccuracyScheduler
from utils.Trainer import Trainer
from utils.EarlyStop import EarlyStopping

from models.bilinear_dense import BilinearDense121, BilinearDense201
from models.bilinear_resnet import BilinearResNet50, BilinearResNet101, BilinearResNet152
from models.bilinear_mobilenet import BilinearMobileNetV2
from models.bilinear_efficientnet import BilinearEfficientnetB0
from models.bilinear_vgg import BilinearVgg16, BilinearVgg19

# 配置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_loader, val_loader, test_loader = train_val_test_data_process()

if __name__ == '__main__':
    opt = parse_opt()

    if opt.net == "bd121":
        model = BilinearDense121(opt.num_classes, opt.attention).to(device)
        model_name = os.path.join("densenet121_fc", str(opt.attention), str(opt.re_train), str(opt.monitor))
        remodel = BilinearDense121(opt.num_classes, opt.attention, None).to(device)
        remodel_name = os.path.join("densenet121_all", str(opt.attention), str(opt.re_train), str(opt.monitor))

    elif opt.net == "bd201":
        model = BilinearDense201(opt.num_classes, opt.attention).to(device)
        model_name = os.path.join("densenet201_fc", str(opt.attention), str(opt.re_train), str(opt.monitor))
        remodel = BilinearDense201(opt.num_classes, opt.attention, None).to(device)
        remodel_name = os.path.join("densenet201_all", str(opt.attention), str(opt.re_train), str(opt.monitor))

    elif opt.net == "br50":
        model = BilinearResNet50(opt.num_classes, opt.attention).to(device)
        model_name = os.path.join("resnet50_fc", str(opt.attention), str(opt.re_train), str(opt.monitor))
        remodel = BilinearResNet50(opt.num_classes, opt.attention, None).to(device)
        remodel_name = os.path.join("resnet50_all", str(opt.attention), str(opt.re_train), str(opt.monitor))

    elif opt.net == "br101":
        model = BilinearResNet101(opt.num_classes, opt.attention).to(device)
        model_name = os.path.join("resnet101_fc", str(opt.attention), str(opt.re_train), str(opt.monitor))
        remodel = BilinearResNet101(opt.num_classes, opt.attention, None).to(device)
        remodel_name = os.path.join("resnet101_all", str(opt.attention), str(opt.re_train), str(opt.monitor))

    elif opt.net == "br152":
        model = BilinearResNet152(opt.num_classes, opt.attention).to(device)
        model_name = os.path.join("resnet152_fc", str(opt.attention), str(opt.re_train), str(opt.monitor))
        remodel = BilinearResNet152(opt.num_classes, opt.attention, None).to(device)
        remodel_name = os.path.join("resnet152_all", str(opt.attention), str(opt.re_train), str(opt.monitor))

    elif opt.net == "be0":
        model = BilinearEfficientnetB0(opt.num_classes, opt.attention).to(device)
        model_name = os.path.join("efficientnet_b0_fc", str(opt.attention), str(opt.re_train), str(opt.monitor))
        remodel = BilinearEfficientnetB0(opt.num_classes, opt.attention, None).to(device)
        remodel_name = os.path.join("efficientnet_b0_all", str(opt.attention), str(opt.re_train), str(opt.monitor))

    elif opt.net == "bm2":
        model = BilinearMobileNetV2(opt.num_classes, opt.attention).to(device)
        model_name = os.path.join("mobilenet_v2_fc", str(opt.attention), str(opt.re_train), str(opt.monitor))
        remodel = BilinearMobileNetV2(opt.num_classes, opt.attention, None).to(device)
        remodel_name = os.path.join("mobilenet_v2_all", str(opt.attention), str(opt.re_train), str(opt.monitor))

    elif opt.net == "bv16":
        model = BilinearVgg16(opt.num_classes, opt.attention).to(device)
        model_name = os.path.join("vgg16_fc", str(opt.attention), str(opt.re_train), str(opt.monitor))
        remodel = BilinearVgg16(opt.num_classes, opt.attention, None).to(device)
        remodel_name = os.path.join("vgg16_all", str(opt.attention), str(opt.re_train), str(opt.monitor))

    elif opt.net == "bv19":
        model = BilinearVgg19(opt.num_classes, opt.attention).to(device)
        model_name = os.path.join("vgg19_fc", str(opt.attention), str(opt.re_train), str(opt.monitor))
        remodel = BilinearVgg19(opt.num_classes, opt.attention, None).to(device)
        remodel_name = os.path.join("vgg19_all", str(opt.attention), str(opt.re_train), str(opt.monitor))

    else:
        model = None
        model_name = None
        remodel = None
        remodel_name = None
        print("print error ,please choose the correct net again")
        exit()

    # 损失
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.fc.parameters(),
                                  lr=opt.lr,
                                  weight_decay=opt.weight_decay)
    lr_scheduler = LRAccuracyScheduler(optimizer, mode='max', patience=5, factor=0.1)
    early_stopping = EarlyStopping(patience=opt.patience / 2, delta=0, monitor=opt.monitor)
    model_path = os.path.join(str(opt.save_model), model_name)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    trainer = Trainer(model, train_loader, val_loader, test_loader, criterion, optimizer, lr_scheduler, early_stopping)
    trainer.train(opt.epochs, model_path)

    if opt.re_train:
        re_optimizer = torch.optim.AdamW(remodel.parameters(),
                                         lr=opt.lr / 10,
                                         weight_decay=opt.weight_decay)
        re_lr_scheduler = LRAccuracyScheduler(re_optimizer, mode='max', patience=5, factor=0.1)
        re_early_stopping = EarlyStopping(patience=opt.patience, delta=0, monitor=opt.monitor)
        remodel_path = os.path.join(str(opt.save_model), remodel_name)
        load_path = os.path.join(str(opt.save_model), model_name) + "/best.pt"
        remodel_state_dict = torch.load(load_path)
        remodel.load_state_dict(remodel_state_dict, strict=False)
        if not os.path.exists(remodel_path):
            os.makedirs(remodel_path)
        trainer = Trainer(remodel, train_loader, val_loader, test_loader, criterion, re_optimizer, re_lr_scheduler,
                          re_early_stopping)
        trainer.train(opt.epochs, remodel_path)

        initial_checkpoint = remodel_path + "/" + os.listdir(remodel_path)[0]
        test_loss, test_acc = trainer.test(test_loader, True, opt.dataset + '/test', initial_checkpoint, remodel_path)
        print('test_acc = ', test_acc)
        print('test_loss = ', test_loss)
    else:
        initial_checkpoint = model_path + "/" + os.listdir(model_path)[0]
        test_loss, test_acc = trainer.test(test_loader, True, opt.dataset + '/test', initial_checkpoint, model_path)
        print('test_acc = ', test_acc)
        print('test_loss = ', test_loss)
