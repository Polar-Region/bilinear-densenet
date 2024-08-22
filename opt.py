# -*- coding: utf-8 -*-
# author --Lee--

import os
import argparse

ROOT = os.getcwd()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--net', type=str, default='bd121', help='choose which net to train')
    parser.add_argument('--attention', type=str, default="CBAM", help='choose which attention to retrain')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0.00001)
    parser.add_argument('--dataset', default=ROOT+"/autodl-tmp/data/")
    parser.add_argument('--save-model', default=ROOT + '/autodl-tmp/runs/save_model/', help='save to project/name')
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--monitor', type=str, default='loss')
    parser.add_argument('--re_train', action=argparse.BooleanOptionalAction)
    parser.add_argument('--log_dir', default="tf-logs/")

    return parser.parse_known_args()[0] if known else parser.parse_args()
