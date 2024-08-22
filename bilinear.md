# bilinear代码详解

## 切分数据集 split

### 切分数据集 `Split.py`

首先在`dataset_dir`和`split_dir`中输入正确的路径，随后在`train_pct,valid_pct,test_pct`中输入切分比例。那么便开始切分数据集为训练集、验证集、测试集。

```python
# 1.确定原图像数据集路径
dataset_dir = "D:/test2021/train_val_test0811/"  # 原始数据集路径
# 2.确定数据集划分后保存的路径
split_dir = "D:/test2021/after0811/"  # 划分后保存路径
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "val")
test_dir = os.path.join(split_dir, "test")
# 3.确定将数据集划分为训练集，验证集，测试集的比例
train_pct = 0.9
valid_pct = 0.1
test_pct = 0
# 4.划分
for root, dirs, files in os.walk(dataset_dir):
    for sub_dir in dirs:  # 遍历0，1，2，3，4，5...9 文件夹
        imgs = os.listdir(os.path.join(root, sub_dir))  # 展示目标文件夹下所有的文件名
        imgs = list(filter(lambda x: x.endswith('.png'), imgs))  # 取到所有以.png结尾的文件，如果改了图片格式，这里需要修改
        random.shuffle(imgs)  # 乱序图片路径
        img_count = len(imgs)  # 计算图片数量
        train_point = int(img_count * train_pct)  # 0:train_pct
        valid_point = int(img_count * (train_pct + valid_pct))  # train_pct:valid_pct

        for i in range(img_count):
            if i < train_point:  # 保存0-train_point的图片到训练集
                out_dir = os.path.join(train_dir, sub_dir)
            elif i < valid_point:  # 保存train_point-valid_point的图片到验证集
                out_dir = os.path.join(valid_dir, sub_dir)
            else:  # 保存valid_point-结束的图片到测试集
                out_dir = os.path.join(test_dir, sub_dir)
            makedir(out_dir)  # 创建文件夹
            target_path = os.path.join(out_dir, os.fsdecode(imgs[i]))  # 指定目标保存路径
            src_path = os.path.join(dataset_dir, os.fsdecode(sub_dir), os.fsdecode(imgs[i]))  # 指定目标原图像路径
            shutil.copy(src_path, target_path)  # 复制图片

        print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point - train_point,img_count - valid_point))
```



## 图片预处理 transform

### 参数提取 `GetStat.py`

 	这个代码主要用处是为了后面图片的归一化参数，在图片归一化中需要图片的**平均值和标准差**

```python
def GetStat(train_data):
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
```

### 预处理 `data_load.py`

 	`train_data_process()`和`test_data_process()`是对训练集和测试集进行`transform`并获得`dataset`和`dataloader`。

```python
# 定义训练集数据处理函数
def train_data_process():
    # 定义训练集的图像变换操作
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=256),  # 调整图像大小为 256 x 256
        torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
        torchvision.transforms.RandomCrop(size=224),  # 随机裁剪成 224 x 224
        torchvision.transforms.ToTensor(),  # 转换为张量
        torchvision.transforms.Normalize(mean, std)  # 归一化
    ])
    # 创建训练集的数据集对象
    train_data = ImageFolder(root=str(opt.dataset) + '/train',  # 设置训练集根目录
                             transform=train_transforms)  # 应用上述变换操作

    # 创建训练集数据加载器
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,  # 设置批次大小
                                               shuffle=True,  # 随机打乱数据集
                                               num_workers=8,  # 设置并行加载的进程数
                                               pin_memory=True)  # 在 CPU 内存中锁定数据，加快 GPU 加载数据的速度
    return train_loader  # 返回数据加载器对象
# 定义测试集数据处理函数
def test_data_process():
    # 定义测试集的图像变换操作
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=256),  # 调整图像大小为 256 x 256
        torchvision.transforms.RandomCrop(size=224),  # 随机裁剪成 224 x 224
        torchvision.transforms.ToTensor(),  # 转换为张量
        torchvision.transforms.Normalize(mean, std)  # 归一化
    ])
    # 创建测试集的数据集对象
    test_data = ImageFolder(root=str(opt.dataset) + '/test',  # 设置测试集根目录
                            transform=test_transforms)  # 应用上述变换操作

    # 创建测试集数据加载器
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=opt.batch_size,  # 设置批次大小
                                              shuffle=False,  # 不打乱数据集
                                              num_workers=8,  # 设置并行加载的进程数
                                              pin_memory=True)  # 在 CPU 内存中锁定数据，加快 GPU 加载数据的速度
    return test_loader  # 返回数据加载器对象
```

## 模型部分 model

 	在这次项目中使用了Dense Net类模型，共有两个模型文件。两个模型文件除了使用的Dense Net类型不同和对应的num_features不同，其他完全一致。在详解中，我们使用`Densenet121`为例。

 	同时在这次项目中加入了注意力机制，此次项目中加入了**CBAM和self-attention**两种简单的注意力机制 

### `AttentionBlock`

 	`class AttentionBlock`是一个添加了自身权重比例的自注意力模型。

```PYTHON
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 计算特征图中每个位置的重要性权重
        weights = self.conv(x)
        weights = f.sigmoid(weights)

        # 对特征图的每个切片进行加权
        x = x * weights

        # 返回加权后的特征图
        return x
```

### `CBAM`

 	`class CBAM`是一个常用的通道注意力、空间注意力的混合注意力机制模型。

```python
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
        spatial_out = self.sigmoid2(spatial_out)

        # 组合通道和空间注意力
        x = x * channel_out * spatial_out

        return x
```

### `BDense121_fc`

 	`Bilinear-Dense121`是利用`DenseNet121`中的`features`作为整个模型的卷积层部分，我们使用两个一模一样的`features`对数据集进行提取特征，随后将得到的特征张量进行**双线性卷积**，随后通过`softmax`对数据集进行分类。

 	`class BDense121_fc`是将`DenseNet121`的`features`部分进行冻结，只训练添加的全连接层部分。

```python
class BDense121_fc(nn.Module):
    def __init__(self, num_classes):
        super(BDense121_fc, self).__init__()
        # 使用DenseNet121的features部分作为卷积层
        self.conv = models.densenet121(weights=
                                       models.DenseNet121_Weights.DEFAULT).features
        # 定义全连接层，输入维度为1024*1024，输出维度为num_classes
        self.fc = nn.Linear(1024 * 1024, num_classes)
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
```

### `BDense121_all`

 	`class BDense121_all`是在`class BDense121_fc`训练完成后，接着对整个模型的所有参数进行训练处理。

```python
class BDense121_all(nn.Module):
    def __init__(self, num_classes):
        super(BDense121_all, self).__init__()
        # 使用DenseNet121的features部分作为卷积层
        self.conv = models.densenet121(weights=None).features
        # 定义全连接层，输入维度为1024*1024，输出维度为num_classes
        self.fc = nn.Linear(1024 * 1024, num_classes)

    def forward(self, x):
        # 前向传播，先经过卷积层，再进行平均池化，最后将输出展平为二维张量
        x = self.conv(x)
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
```

### `BDense121_fc_CBAM`

 	`class BDense121_fc_CBAM`是在`class BDense121_fc`的基础上进行的改进，添加了`CBAM`双通道注意力对输入的特征张量进行注意力处理，更利于进行图像识别。

```python
class BDense121_fc_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(BDense121_fc_CBAM, self).__init__()
        # 使用DenseNet121的features部分作为卷积层
        self.conv = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).features
        # 定义注意力机制层
        self.attention = CBAM(1024)
        # 定义全连接层，输入维度为1024*1024，输出维度为num_classes
        self.fc = nn.Linear(1024 * 1024, num_classes)
        # 将卷积层参数的requires_grad属性设为False，即冻结其参数
        for parameter in self.conv.parameters():
            parameter.requires_grad = False
        # 对全连接层参数进行初始化
        nn.init.kaiming_normal_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias, val=0)

    def forward(self, x):
        # 前向传播，先经过卷积层，再进行平均池化，最后将输出展平为二维张量
        x = self.conv(x)
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
```

## 工具类函数部分 util

### 学习率优化 `LRACC.py`

 	`LRACC.py`是根据模型的准确率来自适应地调整优化器的学习率。

 	在此次代码中：

- 定义学习率调度器，mode为'max'，表示监控最大值
- 当验证集准确率连续5次没有提高，则将学习率乘以0.1

```python
class LRAccuracyScheduler(ReduceLROnPlateau):
    """
    optimizer: 要进行学习率调整的优化器。
    mode: 可以是 'max' 或 'min'，表示要调整的指标是越大越好还是越小越好。
    factor: 学习率调整的因子，新的学习率将是旧的学习率乘以该因子。
    patience: 如果验证集的性能在 patience 个 epoch 内没有改善，则降低学习率。
    threshold: 验证集性能必须提高的阈值，低于该阈值将不会被视为性能提高。
    threshold_mode: 可以是 'rel' 或 'abs'，表示 threshold 是相对于最后一次验证集性能的相对变化还是绝对变化。
    cooldown: 在降低学习率之后，暂停更新学习率的 epoch 数量。
    min_lr: 学习率的下限。
    eps: 用于计算学习率的小数点位数。
    """
    def __init__(self, optimizer, mode='max', factor=0.1, patience=5, threshold=0.0001,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        super(LRAccuracyScheduler, self).__init__(optimizer, mode, factor, patience, threshold,threshold_mode, cooldown, min_lr, eps)

    def step(self, metrics, epoch=None):
        """
        metrics：性能指标，可以是一个数值或者一个元组，其中第二个元素表示准确率；
        epoch：当前的 epoch 数量，可选参数。
        """

        # 如果 metrics 是一个元组，则取其中的第二个元素作为准确率
        if isinstance(metrics, tuple):
            accuracy = metrics[1]
        else:
            accuracy = metrics
        # 获取上一次迭代的学习率
        prev_lr = self.optimizer.param_groups[0]['lr']
        # 调用 ReduceLROnPlateau.step 更新学习率
        super(LRAccuracyScheduler, self).step(accuracy, epoch)
        # 获取新的学习率
        new_lr = self.optimizer.param_groups[0]['lr']
        # 如果学习率发生了变化，则打印一条消息
        if prev_lr != new_lr:
            print(f'学习率从 {prev_lr:.6f} 变为 {new_lr:.6f}。')
```

### 提前停止 `EarlyStop.py`

 	`EarlyStop.py`是当检测的验证集的指标连续没有提升，那么提前停止训练

 	在此次代码中：

- 定义提前停止，连续10次验证集准确率没有提高则停止训练

```python
import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=10, delta=0, monitor='acc'):
        """
        初始化 EarlyStopping 类。

        参数：
        - patience：int，当验证集指标连续多少个 epoch 没有提升时停止训练，默认为 10。
        - delta：float，认为指标提升的最小变化量，默认为 0。
        - monitor：str，要监测的指标名称，可选值为 'acc' 或 'loss'，默认为 'acc'。

        """
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.val_loss_min = np.Inf

    def __call__(self, val, model, path):
        """
        每个 epoch 结束后会调用该方法，根据指标的变化情况决定是否停止训练。

        参数：
        - val：float，当前 epoch 在验证集上的指标值。
        - model：torch.nn.Module，当前的模型。
        - path：str，用于保存模型的路径。

        """
        if self.monitor == 'acc':
            score = val
        else:
            score = -val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val, model, path)
            self.counter = 0

    def save_checkpoint(self, val, model, path):
        """
        保存在验证集上指标最好的模型。

        参数：
        - val：float，当前 epoch 在验证集上的指标值。
        - model：torch.nn.Module，当前的模型。
        - path：str，用于保存模型的路径。

        """
        if self.monitor == 'acc':
            if val > self.val_acc_max:
                torch.save(model.state_dict(), path)
                self.val_acc_max = val
                self.counter = 0
        if self.monitor == 'loss':
            if val < self.val_loss_min:
                torch.save(model.state_dict(), path)
                self.val_loss_min = val
                self.counter = 0
```

### 训练器 `Trainer.py`

#### `Class Trainer`

 	首先，定义了一个训练器类 `Trainer`，包含模型、数据加载器、损失函数、优化器、学习率调整器、设备类型以及提前停止训练实例等参数。

```python
class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, lr_scheduler, early_stopping):
        self.model = model  # 神经网络模型
        self.train_loader = train_loader  # 训练集 DataLoader
        self.test_loader = test_loader  # 测试集 DataLoader
        self.criterion = criterion  # 损失函数
        self.optimizer = optimizer  # 优化器
        self.lr_scheduler = lr_scheduler  # 学习率调整器
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备类型
        self.model.to(self.device)  # 将模型移动到指定的设备上
        self.early_stopping = early_stopping  # 提前停止训练实例
```

#### `Train()`

 	其次设置了`train()`函数，包含了更新学习率、提前停止等模块的函数

```python
    def train(self, num_epochs, model_path):
        start = time.time()  # 记录开始时间
        model_path = model_path + '/best.pt'  # 最佳模型路径
        for epoch in range(num_epochs):  # 遍历每个 epoch
            self.model.train()  # 将模型设置为训练模式
            running_loss = 0.0  # 记录当前 epoch 的总损失
            correct = 0  # 记录当前 epoch 正确分类的数量
            total = 0  # 记录当前 epoch 总共处理的样本数量
            total_step = len(self.train_loader)  # 记录当前 epoch 总共的 batch 数量
            loop = tqdm(enumerate(self.train_loader), total=total_step)  # 创建一个进度条
            for i, (inputs, labels) in loop:  # 遍历每个 batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # 将数据移动到指定的设备上
                self.optimizer.zero_grad()  # 将梯度清零

                outputs = self.model(inputs)  # 前向传播
                loss = self.criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数

                running_loss += loss.item()  # 累计总损失
                _, predicted = torch.max(outputs.data, 1)  # 计算预测结果
                total += labels.size(0)  # 累计样本数量
                correct += (predicted == labels).sum().item()  # 累计正确分类的数量
                running_acc = correct / total  # 当前 epoch 的准确率

                # 更新训练信息
                loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
                loop.set_postfix(loss=loss.item(), acc=running_acc)

            train_loss = running_loss / total_step
            train_acc = correct / total

            # 在测试集上测试模型，得到测试集上的损失和准确率
            test_loss, test_acc = self.test()

            self.lr_scheduler.step(test_acc)  # 更新学习率调度器
            # 根据监控指标进行早停和模型保存
            if opt.monitor == 'acc':
                self.early_stopping(test_acc, self.model, model_path)
            else:
                self.early_stopping(test_loss, self.model, model_path)

            # 打印当前 epoch 的训练和测试信息
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, train_loss, train_acc, test_loss, test_acc))

            # 如果早停法停止训练，则打印提示信息
            if self.early_stopping.early_stop:
                print("Early Stopping")
                break
        end = time.time()  # 记录测试结束时间
        print('train time cost: {:.5f}'.format(end-start))
```

#### `Test()`

 	然后设置`test()`函数，对测试集进行测试，返回模型的测试损失和测试准确率。

 	`test()`主要用于训练过程中的测试，从而查看训练是否存在过拟合，并通过输出进行实时调整。

```python
    def test(self):
        self.model.eval()  # 将模型设置为评估模式
        running_loss = 0.0  # 初始化损失值
        correct = 0  # 初始化正确预测的数量
        total = 0  # 初始化总共的样本数量

        with torch.no_grad():  # 使用 no_grad 上下文管理器，减少计算图的存储，提高代码效率
            for i, (inputs, labels) in enumerate(self.test_loader):  # 遍历测试集数据
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # 将输入和标签转移到 GPU 上

                outputs = self.model(inputs)  # 将输入输入到模型中进行预测
                loss = self.criterion(outputs, labels)  # 计算预测结果与真实标签之间的损失

                running_loss += loss.item()  # 计算总的损失值
                _, predicted = torch.max(outputs.data, 1)  # 找到预测结果中概率最大的类别
                total += labels.size(0)  # 计算总的样本数量
                correct += (predicted == labels).sum().item()  # 计算正确预测的数量

        test_loss = running_loss / len(self.test_loader)  # 计算平均损失
        test_acc = correct / total  # 计算模型在测试集上的准确率

        return test_loss, test_acc  # 返回模型的测试损失和测试准确率
```

#### `Test_Confusion()`

 	最后添加一个`test_confusion`()的函数。`test_confusion()`是一个对测试集进行测试，返回模型的测试损失和测试准确率，同时返回混淆矩阵和混淆矩阵图。

 	`test_confusion()`主要用于训练完成后，对测试及进行测试返回模型的效果参数，并输出混淆矩阵。

```python
    def test_confusion(self, path, initial_checkpoint):
        # 加载模型的参数
        f = torch.load(initial_checkpoint)
        self.model.load_state_dict(f)

        # read class_indict
        labels = os.listdir(path)
        confusion = ConfusionMatrix(num_classes=opt.num_classes, labels=labels)

        start = time.time()
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # 读取类别标签
        labels = os.listdir(path)
        confusion = ConfusionMatrix(num_classes=opt.num_classes, labels=labels)  # 创建混淆矩阵对象

        start = time.time()  # 记录测试的开始时间
        self.model.eval()  # 将模型设置为评估模式
        running_loss = 0.0  # 初始化损失值
        correct = 0  # 初始化正确预测的数量
        total = 0  # 初始化总共的样本数量

        with torch.no_grad():  # 使用 no_grad 上下文管理器，减少计算图的存储，提高代码效率
            for i, (inputs, labels) in enumerate(self.test_loader):  # 遍历测试集数据
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # 将输入和标签转移到 GPU 上

                outputs = self.model(inputs)  # 将输入输入到模型中进行预测
                loss = self.criterion(outputs, labels)  # 计算预测结果与真实标签之间的损失

                running_loss += loss.item()  # 计算总的损失值
                _, predicted = torch.max(outputs.data, 1)  # 找到预测结果中概率最大的类别
                total += labels.size(0)  # 计算总的样本数量
                correct += (predicted == labels).sum().item()  # 计算正确预测的数量

                confusion.update(predicted.cpu().numpy(), labels.cpu().numpy())  # 更新混淆矩阵
        end = time.time()  # 记录测试结束时间
        confusion.plot()  # 绘制混淆矩阵图像
        confusion.summary()  # 输出混淆矩阵的摘要信息
        print("test_confusion time cost: {:.5f} sec".format(end - start))  # 输出测试所花费的时间

        test_loss = running_loss / len(self.test_loader)  # 计算平均损失
        test_acc = correct / total  # 计算模型在测试集上的准确率

        return test_loss, test_acc  # 返回模型的测试损失和测试准确率
```

## 训练和测试部分 Train/Test

### 命令行部分 `opt.py`

这是一个用于解析命令行参数的函数，它使用Python标准库中的`argparse`模块。该函数返回一个`argparse.Namespace`对象，其中包含解析的参数值。解析的参数包括：

- `batch-size`：一个整数，用于指定训练过程中的批次大小。
- `num-classes`：一个整数，表示要分类的类别数量。
- `epochs`：一个整数，表示训练的总轮数。
- `net`：一个字符串，表示要使用的神经网络模型的名称。
- `attention`：一个字符串，表示要使用的注意力机制的类型。
- `lr`：一个浮点数，表示学习率。
- `weight-decay`：一个浮点数，表示权重衰减。
- `dataset`：一个字符串，表示数据集所在的路径。
- `save-model`：一个字符串，表示模型保存的路径。
- `patience`：一个整数，表示早停的耐心值。
- `monitor`：一个字符串，表示用于监视性能的指标，可以是'loss'或'acc'。

```python
import os
import argparse

# 获取当前工作目录的路径，即根目录
ROOT = os.getcwd()


def parse_opt(known=False):
    # 创建一个参数解析器
    parser = argparse.ArgumentParser()

    # 添加一些参数选项
    # 添加一个名为batch-size的命令行参数，表示每个GPU上的批次大小，默认为4
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    # 添加一个名为num-classes的命令行参数，表示要分类的类别数量，默认为46
    parser.add_argument('--num-classes', type=int, default=46)
    # 添加一个名为epochs的命令行参数，表示训练的总轮数，默认为50
    parser.add_argument('--epochs', type=int, default=50)
    # 添加一个名为net的命令行参数，表示要使用的神经网络模型的名称，默认为bdense201_fc
    parser.add_argument('--net', type=str, default='bdense201_fc', help='choose which net to train')
    # 添加一个名为attention的命令行参数，表示要使用的注意力机制的类型，默认为CBAM
    parser.add_argument('--attention', type=str, default='CBAM', help='choose which attention to retrain')
    # 添加一个名为lr的命令行参数，表示学习率，默认为0.0001
    parser.add_argument('--lr', type=float, default=0.0001)
    # 添加一个名为weight-decay的命令行参数，表示权重衰减，默认为0.00001
    parser.add_argument('--weight-decay', type=float, default=0.00001)
    # 添加一个名为dataset的命令行参数，表示数据集所在的路径，默认为当前工作目录下的exp_data目录
    parser.add_argument('--dataset', default=ROOT+"/exp_data/")
    # 添加一个名为save-model的命令行参数，表示模型保存的路径，默认为当前工作目录下的runs/save_model目录
    parser.add_argument('--save-model', default=ROOT + '/runs/save_model/', help='save to project/name')
    # 添加一个名为patience的命令行参数，表示早停的耐心值，默认为10
    parser.add_argument('--patience', type=int, default=10)
    # 添加一个名为monitor的命令行参数，表示用于监视性能的指标，可以是loss或acc，默认为acc
    parser.add_argument('--monitor', type=str, default='acc')

    # 解析命令行参数并返回结果，如果known为True，则返回Namespace对象，否则返回Tuple对象
    return parser.parse_known_args()[0] if known else parser.parse_args()
```

### 训练部分 `train.py`

#### 简单理解

- `criterion` 是损失函数，这里使用了交叉熵损失函数。
- `optimizer` 是优化器，这里使用了 AdamW 优化器，并传入了模型参数、学习率和权重衰减值。
- `lr_scheduler` 是学习率调度器，通过监控验证集准确率来调整学习率，当连续5个`epoch`准确率没有提高时，将学习率乘以0.1。
- `early_stopping` 是提前停止策略，如果验证集准确率在连续的patience次epoch内没有提高，就停止训练。
- `model_path` 是保存模型的路径，如果该路径不存在，则创建一个。
- `trainer` 是一个训练器对象，它封装了训练过程，并将训练结果保存到 `model_path` 中。

#### 特殊之处

- 加入了学习率调度器，连续5个`epoch`准确率没有提高时，将学习率乘以0.1。
- 加入了提前停止策略，如果验证集准确率在连续的`patience`次epoch内没有提高，就停止训练。

```python
# 配置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_loader = train_data_process()
test_loader = test_data_process()

if __name__ == '__main__':
    # 解析命令行参数
    opt = parse_opt()

    # 根据选择的网络和注意力类型选择对应的模型和模型名称
    if opt.net == "bdense121_fc":
        if opt.attention == "CBAM":
            model = BDense121_fc_CBAM(opt.num_classes).to(device)
            model_name = "Densenet121_fc_CBAM/"
            remodel = BDense121_all_CBAM(opt.num_classes).to(device)
            remodel_name = 'Densenet121_all_CBAM/'
        elif opt.attention == "SelfA":
            model = BDense121_fc_Attention(opt.num_classes).to(device)
            model_name = "Densenet121_fc_SelfA/"
            remodel = BDense121_all_Attention(opt.num_classes).to(device)
            remodel_name = 'Densenet121_all_SelfA/'
        else:
            model = BDense121_fc(opt.num_classes).to(device)
            model_name = "Densenet121_fc/"
            remodel = BDense121_all(opt.num_classes).to(device)
            remodel_name = 'Densenet201_all/'

    elif opt.net == "bdense201_fc":
        if opt.attention == "CBAM":
            model = BDense201_fc_CBAM(opt.num_classes).to(device)
            model_name = "Densenet201_fc_CBAM/"
            remodel = BDense201_all_CBAM(opt.num_classes).to(device)
            remodel_name = 'Densenet201_all_CBAM/'
        elif opt.attention == "SelfA":
            model = BDense201_fc_Attention(opt.num_classes).to(device)
            model_name = "Densenet201_fc_SelfA/"
            remodel = BDense201_all_Attention(opt.num_classes).to(device)
            remodel_name = 'Densenet201_all_SelfA/'
        else:
            model = BDense201_fc(opt.num_classes).to(device)
            model_name = "Densenet201_fc/"
            remodel = BDense201_all(opt.num_classes).to(device)
            remodel_name = 'Densenet201_all/'

    else:
        model = None
        model_name = None
        remodel = None
        remodel_name = None
        print("print error ,please choose the correct net again")
        exit()

    # 根据模型类型选择需要训练的部分（全连接层或整个网络）
    if "fc" in model_name:
        op_model = model.fc
    else:
        op_model = model

    # 损失函数为交叉熵
    criterion = nn.CrossEntropyLoss().cuda()

    # 优化器为AdamW，学习率为opt.lr，权重衰减为opt.weight_decay
    optimizer = torch.optim.AdamW(op_model.parameters(),
                                  lr=opt.lr,
                                  weight_decay=opt.weight_decay)

    # 定义学习率调度器，mode为'max'，表示监控最大值
    # 当验证集准确率连续5次没有提高，则将学习率乘以0.1
    lr_scheduler = LRAccuracyScheduler(optimizer, mode='max', patience=5, factor=0.1)

    # 提前停止
    early_stopping = EarlyStopping(patience=opt.patience / 2, delta=0, monitor=opt.monitor)

    # 保存模型路径
    model_path = str(opt.save_model) + model_name

    # 如果模型路径不存在，则创建一个
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # 训练模型
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, lr_scheduler, early_stopping)
    trainer.train(opt.epochs, model_path)
```

### 测试部分 `test.py`

#### 简单理解

 	参数上与`train.py`大致相同，那么test.py的主要用处是将保存后的模型继续测试。并且在测试中可以输出混淆矩阵和对应的图片。

#### 特殊之处

- 我们默认`test.py`是已经训练了两次的模型，所以只支持all类模型的模型文件进行测试。

```python
# *_*coding: utf-8 *_*
# author --Lee--

import torch
import torch.nn as nn

# 解析命令行参数的模块
from opt import parse_opt

# 数据预处理和加载的模块
from data_load import train_data_process, test_data_process

# 工具类模块
from utils.EarlyStop import EarlyStopping  # 提取停止
from utils.LRAcc import LRAccuracyScheduler  # 学习率和准确率的调整器
from utils.Trainer import Trainer  # 训练器

# 加载模型
from models.bilinear_dense201 import BDense201_all
from models.bilinear_dense201 import BDense201_all_Attention
from models.bilinear_dense201 import BDense201_all_CBAM
from models.bilinear_dense121 import BDense121_all
from models.bilinear_dense121 import BDense121_all_Attention
from models.bilinear_dense121 import BDense121_all_CBAM

import warnings


# 配置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_loader = train_data_process()
test_loader = test_data_process()

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    # 解析命令行参数
    opt = parse_opt()

    # 根据选择的网络和注意力类型选择对应的模型和模型名称
    if opt.net == "bdense121_fc":
        if opt.attention == "CBAM":
            model = BDense121_all_CBAM(opt.num_classes).to(device)
        elif opt.attention == "SelfA":
            model = BDense121_all_Attention(opt.num_classes).to(device)
        else:
            model = BDense121_all(opt.num_classes).to(device)

    elif opt.net == "bdense201_fc":
        if opt.attention == "CBAM":
            model = BDense201_all_CBAM(opt.num_classes).to(device)
        elif opt.attention == "SelfA":
            model = BDense201_all_Attention(opt.num_classes).to(device)
        else:
            model = BDense201_all(opt.num_classes).to(device)

        # 损失函数为交叉熵
        criterion = nn.CrossEntropyLoss().cuda()

        # 优化器为AdamW，学习率为opt.lr，权重衰减为opt.weight_decay
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=opt.lr,
                                      weight_decay=opt.weight_decay)

        # 定义学习率调度器，mode为'max'，表示监控最大值
        # 当验证集准确率连续5次没有提高，则将学习率乘以0.1
        lr_scheduler = LRAccuracyScheduler(optimizer, mode='max', patience=5, factor=0.1)

        # 定义提前停止，连续10次验证集准确率没有提高则停止训练
        early_stopping = EarlyStopping(patience=10, delta=0, monitor=opt.monitor)

        # 初始化检查点路径
        initial_checkpoint = '/home/hipeson/Bilinear_Densenet/runs/save_model/Densenet201_all/best.pt'

        # 训练模型
        trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, lr_scheduler, early_stopping)

        # 测试模型并打印测试准确率和损失
        test_loss, test_acc = trainer.test_confusion(opt.dataset, initial_checkpoint)
        print('test_acc = ', test_acc)
        print('test_loss = ', test_loss)
```

### 合成部分 `main.py`

#### 简单理解

`main.py`是结合了`train.py`和`test.py`的代码模型，同时添加了二次训练。

#### 流程解释

1. 首先我们先进行fc类模型的训练，为了防止fc模型训练过拟合，将第一次训练的`epoch`变为了$1/10$，同时将提前停止的`patience`调整为了一半，即连续5次验证集准确率没有提高则停止训练。
2. 随后我们进行all类模型的训练，将`epoch`和`patience`变回初始情况，同时将`lr`设置为初始的$1/10$，从而设置新的优化器和训练器。

#### 特殊之处

- 在第一次训练时，加入了一个识别预训练模型的代码块。若识别到输出的模型文件夹中存在文件，那么将使用该模型进行继续训练。该方法可以有效防止训练中断导致模型无法继续使用。
- 在第一次训练和第二次训练中，我们再次使用了这个方法。我们将fc类模型文件夹中的文件提取，并且作为第二次训练中的预训练模型。

```python
# *_*coding: utf-8 *_*
# author --Lee--

import os
import torch
import torch.nn as nn

# 解析命令行参数的模块
from opt import parse_opt

# 数据预处理和加载的模块
from data_load import train_data_process, test_data_process

# 工具类模块
from utils.EarlyStop import EarlyStopping  # 提取停止
from utils.LRAcc import LRAccuracyScheduler  # 学习率和准确率的调整器
from utils.Trainer import Trainer  # 训练器

# 加载模型
from models.bilinear_dense201 import BDense201_fc, BDense201_all
from models.bilinear_dense201 import BDense201_fc_Attention, BDense201_all_Attention
from models.bilinear_dense201 import BDense201_fc_CBAM, BDense201_all_CBAM
from models.bilinear_dense121 import BDense121_fc, BDense121_all
from models.bilinear_dense121 import BDense121_fc_Attention, BDense121_all_Attention
from models.bilinear_dense121 import BDense121_fc_CBAM, BDense121_all_CBAM

import warnings

# 配置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_loader = train_data_process()
test_loader = test_data_process()

warnings.filterwarnings("ignore")
if __name__ == '__main__':
    # 解析命令行参数
    opt = parse_opt()

    # 根据选择的网络和注意力类型选择对应的模型和模型名称
    if opt.net == "bdense121_fc":
        if opt.attention == "CBAM":
            model = BDense121_fc_CBAM(opt.num_classes).to(device)
            model_name = "Densenet121_fc_CBAM/"
            remodel = BDense121_all_CBAM(opt.num_classes).to(device)
            remodel_name = 'Densenet121_all_CBAM/'
        elif opt.attention == "SelfA":
            model = BDense121_fc_Attention(opt.num_classes).to(device)
            model_name = "Densenet121_fc_SelfA/"
            remodel = BDense121_all_Attention(opt.num_classes).to(device)
            remodel_name = 'Densenet121_all_SelfA/'
        else:
            model = BDense121_fc(opt.num_classes).to(device)
            model_name = "Densenet121_fc/"
            remodel = BDense121_all(opt.num_classes).to(device)
            remodel_name = 'Densenet201_all/'

    elif opt.net == "bdense201_fc":
        if opt.attention == "CBAM":
            model = BDense201_fc_CBAM(opt.num_classes).to(device)
            model_name = "Densenet201_fc_CBAM/"
            remodel = BDense201_all_CBAM(opt.num_classes).to(device)
            remodel_name = 'Densenet201_all_CBAM/'
        elif opt.attention == "SelfA":
            model = BDense201_fc_Attention(opt.num_classes).to(device)
            model_name = "Densenet201_fc_SelfA/"
            remodel = BDense201_all_Attention(opt.num_classes).to(device)
            remodel_name = 'Densenet201_all_SelfA/'
        else:
            model = BDense201_fc(opt.num_classes).to(device)
            model_name = "Densenet201_fc/"
            remodel = BDense201_all(opt.num_classes).to(device)
            remodel_name = 'Densenet201_all/'

    else:
        model = None
        model_name = None
        remodel = None
        remodel_name = None
        print("print error ,please choose the correct net again")
        exit()

    # 损失函数为交叉熵
    criterion = nn.CrossEntropyLoss().cuda()

    # 优化器为AdamW，学习率为opt.lr，权重衰减为opt.weight_decay
    optimizer = torch.optim.AdamW(model.fc.parameters(),
                                  lr=opt.lr,
                                  weight_decay=opt.weight_decay)

    # 定义学习率调度器，mode为'max'，表示监控最大值
    # 当验证集准确率连续5次没有提高，则将学习率乘以0.1
    lr_scheduler = LRAccuracyScheduler(optimizer, mode='max', patience=5, factor=0.1)

    # 定义提前停止，连续5次验证集准确率没有提高则停止训练
    early_stopping = EarlyStopping(patience=opt.patience / 2, delta=0, monitor=opt.monitor)

    # 保存模型路径
    model_path = str(opt.save_model) + model_name

    # 如果模型路径不存在，则创建一个
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if len(os.listdir(model_path)) != 0:
        # 如果之前已经有训练好的模型，则使用第一个模型进行初始化
        initial_checkpoint = model_path + os.listdir(model_path)[0]
        f = torch.load(initial_checkpoint)
        model.load_state_dict(f)

    # 训练模型
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, lr_scheduler, early_stopping)
    trainer.train(int(opt.epochs / 10), model_path)

    # 再次训练
    # 新优化器
    re_optimizer = torch.optim.AdamW(remodel.parameters(),
                                     lr=opt.lr / 10,
                                     weight_decay=opt.weight_decay)

    # 新学习率调度器
    re_lr_scheduler = LRAccuracyScheduler(re_optimizer, mode='max', patience=5, factor=0.1)

    # 定义新提前停止，连续10次验证集准确率没有提高则停止训练
    re_early_stopping = EarlyStopping(patience=opt.patience, delta=0, monitor=opt.monitor)

    # 新保存模型路径
    remodel_path = str(opt.save_model) + remodel_name

    # 如果新保存的模型路径不存在，则创建一个
    if not os.path.exists(remodel_path):
        os.mkdir(remodel_path)

    if len(os.listdir(model_path)) != 0:
        # 使用第一次训练后的模型进行初始化
        initial_checkpoint = model_path + os.listdir(model_path)[0]
        f = torch.load(initial_checkpoint)
        remodel.load_state_dict(f, strict=False)

    # 第二次训练模型
    trainer = Trainer(remodel, train_loader, test_loader, criterion, re_optimizer, re_lr_scheduler, re_early_stopping)
    trainer.train(opt.epochs, remodel_path)

    # 测试第二次训练的模型
    initial_checkpoint = remodel_path + os.listdir(remodel_path)[0]
    test_loss, test_acc = trainer.test_confusion(opt.dataset + '/train', initial_checkpoint)
    print('test_acc = ', test_acc)
    print('test_loss = ', test_loss)
```

# Bilinear-Dense使用方法

## 流程介绍

### 简易流程图

```mermaid
graph LR;
    先进行切分数据集-->图片预处理;
    图片预处理-.bilinear-dense-fc模型.->第一个模型
    第一个模型-.bilinear-dense-all模型.->第二个模型
```

### 具体介绍

1. 首先对数据集进行切分处理，切分为训练集、验证集、测试集。
2. 随后对数据集的图片进行切分、归一化，并且做成`data_loader`
3. 在训练器加入提前停止、学习调度器、使用预训练模型
4. 开始训练并且配置二次训练，随后输出最终模型
5. 对最终模型进行测试，并输出最终准确率和混淆矩阵

## 使用方法

### 安装所需要的库

`epuirment.py`

### 已经切分好的数据集直接进行训练 单卡

```bash
python main.py --num-classes [num_classes] --epochs [epochs] --net [net] --lr [1e-4] --dataset [dataset_root]
```

### 已经切分好的数据集直接进行训练 多卡

暂时不支持多卡



# Bilinear-Dense结果情况

## 总体情况

| 模型选择的/数据集    | 完整数据集 | 鳞翅目数据集 |
| -------------------- | :--------: | :----------: |
| Bilinear-DenseNet121 |   87.78    |    87.00     |
| Bilinear-DenseNet201 |   89.07    |    88.01     |



## Full-Data情况

### Bilinear-DenseNet121

the model accuracy is  0.8778258778258778

|                            | Precision | Recall | Specificity |
| -------------------------- | --------- | ------ | ----------- |
| Abaeis nicippe             | 0.908     | 0.923  | 0.998       |
| Adristyrannus              | 0.966     | 0.737  | 1.0         |
| Ampelophaga                | 0.793     | 0.821  | 0.998       |
| Anthocharis sara           | 0.973     | 0.948  | 1.0         |
| Apolygus lucorum           | 0.971     | 0.919  | 1.0         |
| Army worm                  | 0.485     | 0.505  | 0.992       |
| Ascalapha odorata          | 0.89      | 0.976  | 0.998       |
| Ascia monuste              | 0.908     | 0.868  | 0.998       |
| Asiatic rice borer         | 0.56      | 0.667  | 0.996       |
| Beet army worm             | 0.723     | 0.801  | 0.992       |
| Black cutworm              | 0.842     | 0.871  | 0.996       |
| Bollworm                   | 0.911     | 0.92   | 0.999       |
| Cabbage army worm          | 0.817     | 0.723  | 0.996       |
| Cicadella viridis          | 0.937     | 0.932  | 0.998       |
| Cisseps fulvicollis        | 0.959     | 0.975  | 0.999       |
| Clepsis peritana           | 0.979     | 0.95   | 1.0         |
| Colias eurytheme           | 0.846     | 0.788  | 0.996       |
| Colias philodice           | 0.756     | 0.776  | 0.995       |
| Corn borer                 | 0.797     | 0.849  | 0.993       |
| Deporaus marginatus Pascoe | 0.875     | 0.824  | 1.0         |
| Galgula partita            | 0.938     | 0.946  | 0.999       |
| Grub                       | 0.962     | 0.82   | 1.0         |
| Halysidota tessellaris     | 0.886     | 0.91   | 0.998       |
| Hypena scabra              | 0.933     | 0.917  | 0.999       |
| Hypercompe scribonia       | 0.965     | 0.957  | 0.999       |
| Hypoprepia fucosa          | 0.973     | 0.947  | 1.0         |
| Idia americalis            | 0.972     | 0.937  | 1.0         |
| Lawana imitata Melichar    | 0.757     | 0.737  | 0.999       |
| Lophocampa caryae          | 0.919     | 0.942  | 0.998       |
| Lophocampa maculata        | 0.956     | 0.973  | 0.999       |
| Lycorma delicatula         | 0.856     | 0.884  | 0.993       |
| Meadow moth                | 0.942     | 0.867  | 0.999       |
| Mole cricket               | 0.943     | 0.971  | 0.999       |
| Mythimna unipuncta         | 0.822     | 0.938  | 0.996       |
| Nathalis iole              | 0.908     | 0.929  | 0.998       |
| Noctua pronuba             | 0.787     | 0.856  | 0.995       |
| Oides decempunctata        | 1.0       | 0.946  | 1.0         |
| Papaipema unimodal         | 0.964     | 0.818  | 1.0         |
| Papilio xuthus             | 0.909     | 0.976  | 0.999       |
| Peach borer                | 0.796     | 0.796  | 0.998       |
| Phoebis sennae             | 0.86      | 0.932  | 0.997       |
| Pieris canidia             | 0.893     | 0.812  | 0.998       |
| Pieris marginalis          | 0.871     | 0.918  | 0.998       |
| Pieris rapae               | 0.808     | 0.874  | 0.995       |
| Plutella xylostella        | 0.939     | 0.93   | 0.999       |
| Pontia protodice           | 0.957     | 0.95   | 0.999       |
| Potosiabre vitarsis        | 0.968     | 0.938  | 1.0         |
| Prodenia litura            | 0.736     | 0.779  | 0.992       |
| Pyrisitia lisa             | 0.953     | 0.961  | 0.999       |
| Rhytidodera bowrinii white | 0.968     | 0.769  | 1.0         |
| Rice leaf roller           | 0.92      | 0.947  | 0.998       |
| Rice water weevil          | 0.714     | 0.625  | 0.999       |
| Salurnis marginella Guerr  | 0.86      | 0.86   | 0.999       |
| Spilosoma virginica        | 0.946     | 0.921  | 0.999       |
| Spodoptera ornithogalli    | 0.906     | 0.784  | 0.999       |
| Spoladea recurvalis        | 0.974     | 0.965  | 1.0         |
| Sternochetus frigidus      | 0.882     | 0.682  | 1.0         |
| Wireworm                   | 0.897     | 0.881  | 0.999       |
| Yellow cutworm             | 0.842     | 0.756  | 0.997       |
| Zerene cesonia             | 0.93      | 0.83   | 0.999       |

![matrix](E:\导师工作\实验结果\完整数据集\densenet121\dense121_matrix.png)

### Bilinear-DenseNet201

the model accuracy is  0.890652557319224

|                            | Precision | Recall | Specificity |
| -------------------------- | --------- | ------ | ----------- |
| Abaeis nicippe             | 0.919     | 0.974  | 0.998       |
| Adristyrannus              | 0.879     | 0.763  | 0.999       |
| Ampelophaga                | 0.758     | 0.839  | 0.998       |
| Anthocharis sara           | 0.948     | 0.948  | 0.999       |
| Apolygus lucorum           | 0.958     | 0.932  | 1.0         |
| Army worm                  | 0.635     | 0.581  | 0.995       |
| Ascalapha odorata          | 0.952     | 0.968  | 0.999       |
| Ascia monuste              | 0.875     | 0.86   | 0.998       |
| Asiatic rice borer         | 0.565     | 0.619  | 0.997       |
| Beet army worm             | 0.785     | 0.813  | 0.994       |
| Black cutworm              | 0.872     | 0.884  | 0.997       |
| Bollworm                   | 0.888     | 0.87   | 0.998       |
| Cabbage army worm          | 0.831     | 0.73   | 0.996       |
| Cicadella viridis          | 0.963     | 0.948  | 0.999       |
| Cisseps fulvicollis        | 0.975     | 0.975  | 1.0         |
| Clepsis peritana           | 0.99      | 0.95   | 1.0         |
| Colias eurytheme           | 0.798     | 0.812  | 0.995       |
| Colias philodice           | 0.791     | 0.75   | 0.996       |
| Corn borer                 | 0.839     | 0.876  | 0.995       |
| Deporaus marginatus Pascoe | 0.923     | 0.706  | 1.0         |
| Galgula partita            | 0.956     | 0.964  | 0.999       |
| Grub                       | 0.962     | 0.836  | 1.0         |
| Halysidota tessellaris     | 0.92      | 0.937  | 0.999       |
| Hypena scabra              | 0.918     | 0.926  | 0.998       |
| Hypercompe scribonia       | 0.965     | 0.957  | 0.999       |
| Hypoprepia fucosa          | 0.991     | 0.956  | 1.0         |
| Idia americalis            | 0.955     | 0.955  | 0.999       |
| Lawana imitata Melichar    | 0.789     | 0.789  | 0.999       |
| Lophocampa caryae          | 0.908     | 0.983  | 0.998       |
| Lophocampa maculata        | 0.982     | 0.964  | 1.0         |
| Lycorma delicatula         | 0.85      | 0.887  | 0.993       |
| Meadow moth                | 0.952     | 0.885  | 0.999       |
| Mole cricket               | 0.971     | 0.98   | 1.0         |
| Mythimna unipuncta         | 0.886     | 0.965  | 0.998       |
| Nathalis iole              | 0.951     | 0.913  | 0.999       |
| Noctua pronuba             | 0.796     | 0.904  | 0.995       |
| Oides decempunctata        | 1.0       | 0.946  | 1.0         |
| Papaipema unimodal         | 0.967     | 0.879  | 1.0         |
| Papilio xuthus             | 0.975     | 0.951  | 1.0         |
| Peach borer                | 0.833     | 0.833  | 0.999       |
| Phoebis sennae             | 0.896     | 0.909  | 0.998       |
| Pieris canidia             | 0.919     | 0.857  | 0.998       |
| Pieris marginalis          | 0.835     | 0.918  | 0.997       |
| Pieris rapae               | 0.864     | 0.896  | 0.997       |
| Plutella xylostella        | 0.959     | 0.94   | 0.999       |
| Pontia protodice           | 0.951     | 0.986  | 0.999       |
| Potosiabre vitarsis        | 0.857     | 0.938  | 0.999       |
| Prodenia litura            | 0.687     | 0.791  | 0.99        |
| Pyrisitia lisa             | 0.992     | 0.977  | 1.0         |
| Rhytidodera bowrinii white | 0.944     | 0.872  | 1.0         |
| Rice leaf roller           | 0.94      | 0.947  | 0.999       |
| Rice water weevil          | 0.75      | 0.625  | 0.999       |
| Salurnis marginella Guerr  | 0.881     | 0.86   | 0.999       |
| Spilosoma virginica        | 0.963     | 0.912  | 0.999       |
| Spodoptera ornithogalli    | 0.874     | 0.811  | 0.998       |
| Spoladea recurvalis        | 0.965     | 0.957  | 0.999       |
| Sternochetus frigidus      | 0.867     | 0.591  | 1.0         |
| Wireworm                   | 0.915     | 0.915  | 0.999       |
| Yellow cutworm             | 0.855     | 0.787  | 0.997       |
| Zerene cesonia             | 0.917     | 0.884  | 0.999       |

![](E:\导师工作\实验结果\完整数据集\densenet201\dense201_matrix.png)

## Exp-Data 鳞翅目情况

### Bilinear-DenseNet121

test_acc =  0.8699980853915374
test_loss =  0.42143810437940293

|                         | Precision | Recall | Specificity |
| ----------------------- | --------- | ------ | ----------- |
| Abaeis nicippe          | 0.924     | 0.932  | 0.998       |
| Adristyrannus           | 0.879     | 0.763  | 0.999       |
| Ampelophaga             | 0.88      | 0.786  | 0.999       |
| Anthocharis sara        | 0.955     | 0.93   | 0.999       |
| Army worm               | 0.52      | 0.559  | 0.991       |
| Ascalapha odorata       | 0.96      | 0.968  | 0.999       |
| Ascia monuste           | 0.829     | 0.807  | 0.996       |
| Asiatic rice borer      | 0.56      | 0.667  | 0.996       |
| Beet army worm          | 0.752     | 0.729  | 0.992       |
| Black cutworm           | 0.849     | 0.878  | 0.995       |
| Bollworm                | 0.835     | 0.91   | 0.996       |
| Cabbage army worm       | 0.724     | 0.743  | 0.992       |
| Cisseps fulvicollis     | 0.975     | 0.975  | 0.999       |
| Clepsis peritana        | 0.979     | 0.92   | 1.0         |
| Colias eurytheme        | 0.797     | 0.712  | 0.994       |
| Colias philodice        | 0.621     | 0.75   | 0.99        |
| Corn borer              | 0.84      | 0.881  | 0.994       |
| Galgula partita         | 0.972     | 0.946  | 0.999       |
| Halysidota tessellaris  | 0.963     | 0.928  | 0.999       |
| Hypena scabra           | 0.912     | 0.942  | 0.998       |
| Hypercompe scribonia    | 0.982     | 0.966  | 1.0         |
| Hypoprepia fucosa       | 0.991     | 0.965  | 1.0         |
| Idia americalis         | 0.973     | 0.964  | 0.999       |
| Lophocampa caryae       | 0.934     | 0.934  | 0.998       |
| Lophocampa maculata     | 0.939     | 0.964  | 0.999       |
| Meadow moth             | 0.934     | 0.876  | 0.999       |
| Mythimna unipuncta      | 0.897     | 0.929  | 0.998       |
| Nathalis iole           | 0.891     | 0.906  | 0.997       |
| Noctua pronuba          | 0.848     | 0.896  | 0.996       |
| Papaipema unimodal      | 0.963     | 0.788  | 1.0         |
| Papilio xuthus          | 0.976     | 0.976  | 1.0         |
| Peach borer             | 0.909     | 0.741  | 0.999       |
| Phoebis sennae          | 0.921     | 0.879  | 0.998       |
| Pieris canidia          | 0.886     | 0.82   | 0.997       |
| Pieris marginalis       | 0.8       | 0.873  | 0.995       |
| Pieris rapae            | 0.806     | 0.83   | 0.995       |
| Plutella xylostella     | 0.876     | 0.92   | 0.997       |
| Pontia protodice        | 0.923     | 0.942  | 0.998       |
| Prodenia litura         | 0.736     | 0.779  | 0.99        |
| Pyrisitia lisa          | 0.904     | 0.961  | 0.997       |
| Rice leaf roller        | 0.92      | 0.955  | 0.998       |
| Spilosoma virginica     | 0.948     | 0.956  | 0.999       |
| Spodoptera ornithogalli | 0.872     | 0.856  | 0.997       |
| Spoladea recurvalis     | 0.956     | 0.948  | 0.999       |
| Yellow cutworm          | 0.827     | 0.717  | 0.996       |
| Zerene cesonia          | 0.926     | 0.786  | 0.999       |

![matrix](E:\导师工作\实验结果\鳞翅目数据集\densenet121\matrix.png)

### Bilinear-DenseNet201

test_acc =  0.8811028144744399

test_loss =  0.4531798364411041

|                         | Precision | Recall | Specificity |
| ----------------------- | --------- | :----- | ----------- |
| Abaeis nicippe          | 0.965     | 0.94   | 0.999       |
| Adristyrannus           | 0.886     | 0.816  | 0.999       |
| Ampelophaga             | 0.797     | 0.839  | 0.998       |
| Anthocharis sara        | 0.982     | 0.939  | 1.0         |
| Army worm               | 0.543     | 0.538  | 0.992       |
| Ascalapha odorata       | 0.967     | 0.96   | 0.999       |
| Ascia monuste           | 0.896     | 0.833  | 0.998       |
| Asiatic rice borer      | 0.6       | 0.714  | 0.996       |
| Beet army worm          | 0.794     | 0.765  | 0.993       |
| Black cutworm           | 0.894     | 0.857  | 0.997       |
| Bollworm                | 0.863     | 0.88   | 0.997       |
| Cabbage army worm       | 0.766     | 0.75   | 0.993       |
| Cisseps fulvicollis     | 0.967     | 0.975  | 0.999       |
| Clepsis peritana        | 0.941     | 0.95   | 0.999       |
| Colias eurytheme        | 0.781     | 0.738  | 0.993       |
| Colias philodice        | 0.691     | 0.733  | 0.993       |
| Corn borer              | 0.862     | 0.876  | 0.995       |
| Galgula partita         | 0.955     | 0.946  | 0.999       |
| Halysidota tessellaris  | 0.961     | 0.892  | 0.999       |
| Hypena scabra           | 0.904     | 0.934  | 0.998       |
| Hypercompe scribonia    | 0.943     | 0.991  | 0.999       |
| Hypoprepia fucosa       | 0.982     | 0.947  | 1.0         |
| Idia americalis         | 0.955     | 0.964  | 0.999       |
| Lophocampa caryae       | 0.906     | 0.959  | 0.998       |
| Lophocampa maculata     | 0.946     | 0.946  | 0.999       |
| Meadow moth             | 0.935     | 0.885  | 0.999       |
| Mythimna unipuncta      | 0.864     | 0.956  | 0.997       |
| Nathalis iole           | 0.944     | 0.937  | 0.999       |
| Noctua pronuba          | 0.855     | 0.896  | 0.996       |
| Papaipema unimodal      | 0.966     | 0.848  | 1.0         |
| Papilio xuthus          | 0.93      | 0.976  | 0.999       |
| Peach borer             | 0.957     | 0.833  | 1.0         |
| Phoebis sennae          | 0.937     | 0.894  | 0.998       |
| Pieris canidia          | 0.868     | 0.842  | 0.997       |
| Pieris marginalis       | 0.787     | 0.909  | 0.995       |
| Pieris rapae            | 0.863     | 0.837  | 0.996       |
| Plutella xylostella     | 0.95      | 0.95   | 0.999       |
| Pontia protodice        | 0.938     | 0.971  | 0.998       |
| Prodenia litura         | 0.773     | 0.814  | 0.992       |
| Pyrisitia lisa          | 0.926     | 0.977  | 0.998       |
| Rice leaf roller        | 0.915     | 0.97   | 0.998       |
| Spilosoma virginica     | 0.954     | 0.904  | 0.999       |
| Spodoptera ornithogalli | 0.902     | 0.829  | 0.998       |
| Spoladea recurvalis     | 0.948     | 0.957  | 0.999       |
| Yellow cutworm          | 0.773     | 0.78   | 0.994       |
| Zerene cesonia          | 0.877     | 0.83   | 0.997       |

![](E:\导师工作\实验结果\鳞翅目数据集\densenet201\matrix.png)

# Bilinear-Dense存在问题

## 数据集对应的问题

Army worm、Asiatic rice borer、Lawana imitata Melichar、Rice water weevil的准确率普遍偏低。

1. Army worm、Asiatic rice borer都有幼虫和蛾类的图像
2. Lawana imitata Melichar是模样很像绿色叶子
3. Rice water weevil的感觉是种类偏多，而且拍照角度各异

## 模型文件对应的问题

1.  初始模型所需要的显存要求较高，模型较为复杂。
2. 在加入了注意力机制后，增加了更多的参数，对于难以的识别的更易于进行图像识别，但是相比所需要的时间和显存也更高。
3. 同时，两次训练非常容易导致过拟合。本身加入了注意力机制后的代码识别能力较强，所以对应导致的代码的过拟合能力较严重。
