#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
 @Author: HUJING
 @Time:5/26/18 11:11 AM 2018
 @Email: jhsa26@mail.ustc.edu.cn
 @Site:jhsa26.github.io
 """
import torch.nn as nn
import torch
from torch import autograd

#把常用的2个卷积操作简单封装下
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), #添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch,image_len,image_len_out):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 8)
        # self.pool1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.conv2 = DoubleConv(8, 16)
        # self.pool2 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.conv3 = DoubleConv(16, 32)
        # self.pool3 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.conv4 = DoubleConv(32, 64)
        # self.pool4 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.conv5 = DoubleConv(64, 128)
        # 逆卷积，也可以使用上采样
        self.up6 = nn.ConvTranspose2d(128, 64, kernel_size=3,stride=1,padding=1)
        self.conv6 = DoubleConv(128, 64)
        self.up7 = nn.ConvTranspose2d(64,32, kernel_size=3,stride=1,padding=1)
        self.conv7 = DoubleConv(64,32)
        self.up8 = nn.ConvTranspose2d(32, 16, kernel_size=3,stride=1,padding=1)
        self.conv8 = DoubleConv(32, 16)
        self.up9 = nn.ConvTranspose2d(16,8, kernel_size=3,stride=1,padding=1)
        self.conv9 = DoubleConv(16,8)
        # self.conv10 = nn.Conv2d(8, out_ch, kernel_size=3,stride=1,padding=1)
        self.conv10 = nn.Conv2d(8, out_ch, kernel_size=(1,5), stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=out_ch*image_len, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=image_len_out)
        self.dropout = nn.Dropout2d(p=0.1)
    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.dropout(c1)
        # p1 = self.pool1(c1)
        c2 = self.conv2(c1)
        c2 = self.dropout(c2)
        # p2 = self.pool2(c2)
        c3 = self.conv3(c2)
        c3 = self.dropout(c3)
        # p3 = self.pool3(c3)
        c4 = self.conv4(c3)
        c4 = self.dropout(c4)
        # p4 = self.pool4(c4)
        c5 = self.conv5(c4)
        c6 = self.dropout(c5)

        up_6 = self.up6(c5)

        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        c6 = self.dropout(c6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        c7 = self.dropout(c7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        c8 = self.dropout(c8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c9 = self.dropout(c9)
        c10 = self.conv10(c9)
        c10 = self.dropout(c10)
        out = c10
        out = out.view([out.size(0), out.size(2), out.size(3)])
        # out = c10.view(c10.size(0), -1)
        # out = self.fc1(out)
        # out = self.fc2(out)
        return out
if __name__ == '__main__':
    pass