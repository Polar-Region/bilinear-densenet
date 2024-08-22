# -*- coding: utf-8 -*-
# author --Lee--

import time

from models.bilinear_dense import BilinearDense121, BilinearDense201
from torchsummary import summary

from opt import parse_opt
opt = parse_opt()


# net = BilinearDense121(opt.num_classes)
net = BilinearDense201(opt.num_classes, opt.attention)

start = time.time()
in_channel = net.fc.in_features
summary(net, (3, 224, 224))
end = time.time()
print(end-start)
