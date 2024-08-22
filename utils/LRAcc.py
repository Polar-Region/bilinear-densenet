from torch.optim.lr_scheduler import ReduceLROnPlateau


class LRAccuracyScheduler(ReduceLROnPlateau):
    def __init__(self, optimizer, mode='max', factor=0.1, patience=5, threshold=0.0001,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        super(LRAccuracyScheduler, self).__init__(optimizer, mode, factor, patience, threshold,
                                                  threshold_mode, cooldown, min_lr, eps)

    def step(self, metrics, epoch=None):
        if isinstance(metrics, tuple):
            accuracy = metrics[1]
        else:
            accuracy = metrics
        prev_lr = self.optimizer.param_groups[0]['lr']  # get previous learning rate
        super(LRAccuracyScheduler, self).step(accuracy, epoch)  # update learning rate
        new_lr = self.optimizer.param_groups[0]['lr']  # get new learning rate
        if prev_lr != new_lr:
            print(f'Learning rate changed from {prev_lr:.6f} to {new_lr:.6f}.')  # print message
