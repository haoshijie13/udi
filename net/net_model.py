import os
import sys
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__)))
import net_utils as utils


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = utils.DoubleConv(n_channels, 64)
        self.down1 = utils.Down(64, 128)
        self.down2 = utils.Down(128, 256)
        self.down3 = utils.Down(256, 512)
        self.down4 = utils.Down(512, 1024)
        self.up1 = utils.Up(1024, 512, bilinear)
        self.up2 = utils.Up(512, 256, bilinear)
        self.up3 = utils.Up(256, 128, bilinear)
        self.up4 = utils.Up(128, 64, bilinear)
        self.outc = utils.OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)

