import os

import numpy as np
import torch
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benckmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logging(object):

    def __init__(self, log_dir=""):
        if not os.path.exists(log_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(log_dir)

        self.log_file = os.path.join(log_dir, 'logging.txt')

    def __call__(self, line='', *args, **kwargs):
        with open(self.log_file, 'a+', encoding="utf-8") as f:
            f.writelines(line + '\n')
        print(line)
def rand_bbox(H,W, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
if __name__ == '__main__':
    logger=Logging('./logging/PMT')
    logger(line='xxxx')