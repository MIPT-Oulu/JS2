"Based on https://github.com/MIPT-Oulu/Oulu-MIPT-ML-Seminar-2018/tree/master/Tutorial_2/codes"

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True


def train_epoch(epoch, net, optimizer, train_loader, criterion):
    net.train(True)
    running_loss = 0.0
    n_batches = len(train_loader)
    pbar = tqdm(total=n_batches)

    for i, (images, jsw_des, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # forward + backward + optimize
        images = images.to(device)
        jsw_des = jsw_des.to(device)

        labels = labels.float().to(device)
        outputs = net(images, jsw_des).squeeze()

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_description('Train loss: %.3f / loss %.3f' % (running_loss / (i + 1), loss.item()))
        pbar.update()
    pbar.close()
    return running_loss / n_batches


def validate_epoch(net, val_loader, criterion):
    probs_lst = []
    ground_truth = []
    net.eval()

    running_loss = 0.0
    n_batches = len(val_loader)
    sm = nn.Sigmoid()

    for i, (images, jsw, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.float().to(device)
        jsw = jsw.to(device)

        outputs = net(images,jsw).squeeze()
        loss = criterion(outputs, labels)

        targets = labels.cpu().numpy()
        preds = sm(outputs).data.cpu().numpy()

        probs_lst.append(preds)
        ground_truth.append(targets)

        running_loss += loss.item()


    probs_lst = np.hstack(probs_lst)
    ground_truth = np.hstack(ground_truth)

    return running_loss / n_batches, probs_lst, ground_truth


def adjust_learning_rate(optimizer, epoch, lr_min, init_lr, lr_drop):
    """
    Decreases the initial LR by 5 every drop_step epochs.
    Conv layers learn slower if specified in the optimizer.
    """
    lr = init_lr * (0.4 ** (epoch // lr_drop))
    if lr < lr_min:
        lr = lr_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr
