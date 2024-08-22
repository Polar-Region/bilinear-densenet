import os

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from opt import parse_opt

opt = parse_opt()


class ConfusionMatrix(object):

    def __init__(self, num_classes, labels, path):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.path = path

    def update(self, prediction, labels):
        for p, t in zip(prediction, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_tp = 0
        for i in range(self.num_classes):
            sum_tp += self.matrix[i, i]
        acc = sum_tp / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        save_txt_path = os.path.join(self.path, 'confusion_matrix.txt')
        np.savetxt(save_txt_path, matrix, delimiter=",", fmt='%d')  # 整数
        classes = []
        for i in range(opt.num_classes):
            classes.append(i)
        proportion = []
        length = len(matrix)
        for i in matrix:
            for j in i:
                temp = j / (np.sum(i))
                proportion.append(temp)
        pshow = []
        for i in proportion:
            pt = "%.2f%%" % (i * 100)
            pshow.append(pt)
        proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
        plt.figure(figsize=(7, 7))
        plt.ylabel('True label', fontsize=10)
        plt.xlabel('Predict label', fontsize=10)
        plt.xticks(np.arange(0, len(classes) + 1, step=5), fontsize=6)
        plt.yticks(np.arange(0, len(classes) + 1, step=5), fontsize=6)
        plt.title('matrix')
        ax = plt.gca()
        # 去掉边框
        # ax.spines['top'].set_color('none')
        # ax.spines['right'].set_color('none')
        # 移位置 设为原点相交
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))
        ax.yaxis.set_ticks_position('right')
        ax.spines['right'].set_position(('data', 0))
        plt.xlim(0, len(classes))
        plt.ylim(0, len(classes))

        plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
        # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
        # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
        plt.colorbar()
        plt.tight_layout()
        save_path = os.path.join(self.path, 'matrix.png')
        plt.savefig(save_path)
        plt.show()
