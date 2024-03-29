import os
import time

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from dataset import Loader
from net_model import UNet
from net_utils import FocalLoss


def train_net(net, device, data_path, epochs=2000, batch_size=1, lr=0.001, resume=False):
    dataset = Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1E-8, momentum=0.9)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(1, np.ceil(epochs)/20) * 20, gamma=0.85)

    start_epoch = 0
    tot_loss = []
    if resume:
        ckpt = torch.load('checkpoint.pth')
        net.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch'] + 1
        lr_schedule.load_state_dict(ckpt['lr_schedule'])
        tot_loss = ckpt['tot_loss']

    criterion = FocalLoss(logits=True)
    # criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    best_acc = 0
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        net.train()
        i = 0
        num = 0
        step = list(np.round(np.linspace(0, len(train_loader), num=21)))
        loss_lst = np.zeros(len(train_loader))
        accuracy_lst = np.zeros(len(train_loader))

        for img, lbl in train_loader:
            ts = time.time()

            optimizer.zero_grad()
            img = img.to(device=device, dtype=torch.float32)
            lbl = lbl.to(device=device, dtype=torch.float32)
            pred = net(img)

            loss = criterion(pred, lbl) * 1000
            loss_lst[i] = loss
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')

            pred = np.array(pred.data.cpu()[0])[0]
            lbl = np.array(lbl.data.cpu()[0])[0]
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc_map = pred - lbl
            accuracy = acc_map[acc_map == 0].size / acc_map.size
            accuracy_lst[i] = accuracy
            if accuracy > best_acc:
                best_acc = accuracy

            loss.backward()
            optimizer.step()

            i += 1
            if i in step:
                num = step.index(i)

            te = time.time()

            print('Epoch {}: [{}{}] {}%, loss: {}, accuracy: {}, lr: {}, cost: {}s'.format(epoch, '█' * num, '-' * (20 - num), num * 5, loss, accuracy, optimizer.state_dict()['param_groups'][0]['lr'], te - ts))

        lr_schedule.step()

        tot_loss.extend(list(loss_lst))

        if not os.path.exists('figure'):
            os.mkdir('figure')

        plt.clf()
        plt.plot(np.arange(len(train_loader) * (epoch+1)), tot_loss)
        plt.xticks([])
        plt.savefig(os.path.join('figure', 'Total_loss.png'.format(epoch)))

        plt.clf()
        plt.plot(np.arange(len(train_loader)), loss_lst)
        plt.xticks([])
        plt.savefig(os.path.join('figure', 'Epoch_{}_loss.png'.format(epoch)))

        plt.clf()
        plt.plot(np.arange(len(train_loader)), accuracy_lst, 'r')
        plt.xticks([])
        plt.savefig(os.path.join('figure', 'Epoch_{}_accuracy.png'.format(epoch)))

        if epoch % 10 == 0:
            checkpoint = {'net': net.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch, 'lr_schedule': lr_schedule.state_dict(), 'tot_loss': tot_loss}
            torch.save(checkpoint, 'checkpoint.pth')

        t1 = time.time()

        print('\nEpoch {} cost: {}, minimum loss: {}, maximum accuracy: {}'.format(epoch, t1 - t0, np.min(loss_lst), np.max(accuracy_lst)))
        print('Best loss: {}, best accuracy: {}\n'.format(best_loss, best_acc))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)

    data_path = r'D:\test_data\singlecell\RNA_Segment'
    train_net(net, device, data_path, resume=True)

