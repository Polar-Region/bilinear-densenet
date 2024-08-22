import torch
import time
import os
from tqdm import tqdm

from utils.Confusion import ConfusionMatrix
from torch.utils.tensorboard import SummaryWriter

from opt import parse_opt

opt = parse_opt()


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 criterion, optimizer, lr_scheduler, early_stopping):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.early_stopping = early_stopping

        # 初始化一个用于记录的 SummaryWriter
        log_dir = os.path.join(opt.log_dir, opt.net, str(opt.attention), str(opt.re_train))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(str(log_dir))

    def train(self, num_epochs, model_path):
        start = time.time()
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            total_step = len(self.train_loader)
            # 创建一个进度条，并设置总共的step数量
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for i, (inputs, labels) in loop:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_acc = correct / total

                # 更新训练信息
                loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
                loop.set_postfix(loss=loss.item(), acc=running_acc)

            train_loss = running_loss / total_step
            train_acc = correct / total

            val_loss, val_acc = self.test(self.val_loader)

            self.lr_scheduler.step(val_acc)
            if opt.monitor == 'acc':
                self.early_stopping(val_acc, self.model, model_path + '/best.pt')
            else:
                self.early_stopping(val_loss, self.model, model_path + '/best.pt')

            print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))

            # 将验证损失和准确性记录到 TensorBoard
            self.writer.add_scalar('Validation/Loss', val_loss, epoch)
            self.writer.add_scalar('Validation/Accuracy', val_acc, epoch)

            if self.early_stopping.early_stop:
                print("Early Stopping")
                self.writer.add_text('Event', 'Early Stopping', global_step=epoch)
                break
        end = time.time()
        print('train time cost: {:.5f}'.format(end - start))

    def test(self, loader, confusion=None, path=None, initial_checkpoint=None, save_path=None):
        if confusion:
            f = torch.load(initial_checkpoint)
            self.model.load_state_dict(f)

            # read class_indict
            labels = os.listdir(path)
            confusion = ConfusionMatrix(num_classes=opt.num_classes, labels=labels, path=save_path)
        start = time.time()
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                confusion.update(predicted.cpu().numpy(), labels.cpu().numpy()) if confusion else None
        if confusion:
            end = time.time()
            confusion.plot()
            confusion.summary()
            print("test_confusion time cost: {:.5f} sec".format(end - start))

        test_loss = running_loss / len(loader)
        test_acc = correct / total

        # 记录测试损失和准确性到 TensorBoard
        self.writer.add_scalar('Test/Loss', test_loss, 0)  # 为简便起见，将步骤设为 0
        self.writer.add_scalar('Test/Accuracy', test_acc, 0)  # 为简便起见，将步骤设为 0

        # 在训练结束后添加以下代码，用于将结果写入文本文件
        if str(loader) == "test_loader":
            with open('test_results.txt', 'a') as f:
                f.write(f"net: {opt.net},attention:{opt.attention},monitor:{opt.monitor}, re_train:{opt.re_train} "
                        f"accuracy: {test_acc}, loss: {test_loss}\n")

        # 上述代码将网络名称、最佳准确度和最佳损失写入名为 'test_results.txt' 的文本文件。

        return test_loss, test_acc
