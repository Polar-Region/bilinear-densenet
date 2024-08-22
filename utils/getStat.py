import torch
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms

import opt

opt = opt.parse_opt()


def get_stat(train_data):
    # 计算训练数据集的平均值和标准差
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            # 对每个通道（RGB）计算平均值和标准差
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    # 对每个通道的平均值和标准差进行平均，得到数据集的平均值和标准差
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())
 
 
if __name__ == '__main__':
    # 加载训练数据集并计算平均值和标准差
    train_dataset = ImageFolder(root=str(opt.dataset)+'/train', transform=transforms.ToTensor())
    print(get_stat(train_dataset))
