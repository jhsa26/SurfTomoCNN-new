#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
 @Author: HUJING
 @Time:5/14/18 11:18 AM 2018
 @Email: jhsa26@mail.ustc.edu.cn
 @Site:jhsa26.github.io
 """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# conv
# W_{n+1} = (W_{n} + Padding*2 - K_{w})/S + 1
# H_{n+1} = (H_{n} + Padding*2 - K_{h})/S + 1
# pooling
# W{n+1} = (W{n} - K{w})/S+1
# H{n+1} = (H{n} - K{h})/S
class Net(nn.Module):
    def __init__(self,image_width,
                 image_height,
                 image_outwidth,
                 image_outheight,
                 inchannel,
                 outchannel=4):
        super(Net,self).__init__()
        self.width = image_outwidth
        self.height = image_outheight
        self.conv1 = nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=3,stride=1,padding=1)  # 1 to 10
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(outchannel)

        self.conv2 = nn.Conv2d(in_channels=outchannel,out_channels=outchannel*2,kernel_size=3,stride=1,padding=1) # 10 to 20
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(outchannel*2)
	#outchannel = outchannel*2
        self.conv3 = nn.Conv2d(in_channels=outchannel*2,out_channels=outchannel*4,kernel_size=3,stride=1,padding=1) # 10 to 20
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(outchannel*4)
	#outchannel = outchannel*2
        self.conv4 = nn.Conv2d(in_channels=outchannel*4,out_channels=outchannel*4,kernel_size=3,stride=1,padding=1) # 10 to 20
        self.pool4 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(outchannel*4)
	#outchannel = outchannel*2
        # self.conv5 = nn.Conv2d(in_channels=outchannel*8,out_channels=outchannel*8,kernel_size=3,stride=1,padding=1) # 10 to 20
        # self.pool5 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        # self.bn5 = nn.BatchNorm2d(outchannel*8)
        #
        # self.conv6 = nn.Conv2d(in_channels=outchannel*4,out_channels=outchannel*4,kernel_size=3,stride=1,padding=1) # 10 to 20
        # self.pool6 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        # self.bn6 = nn.BatchNorm2d(outchannel*4)
        #
        # self.conv7 = nn.Conv2d(in_channels=outchannel*4,out_channels=outchannel*8,kernel_size=3,stride=1,padding=1) # 10 to 20
        # self.pool7 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        # self.bn7 = nn.BatchNorm2d(outchannel*8)
        #
        # self.conv8 = nn.Conv2d(in_channels=outchannel*8,out_channels=outchannel*8,kernel_size=3,stride=1,padding=1) # 10 to 20
        # self.pool8 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        # self.bn8 = nn.BatchNorm2d(outchannel*8)



        self.fc1 = nn.Linear(in_features=outchannel*4*image_height*image_width, out_features=image_outheight*image_outwidth)
        # self.fc2 = nn.Linear(in_features=128, out_features=64)
        # self.fc3 = nn.Linear(in_features=64, out_features=13)
        # self.fc4 = nn.Linear(in_features=32, out_features=13)
        self.dropout = nn.Dropout2d(p=0.2)
        # self.adaptpool = nn.AdaptiveMaxPool2d((1,21)) # fix the number of frequencies as 21.
    def forward(self, input):




        x = F.leaky_relu((self.conv1(input)))   #self.pool1(F.relu(self.bn1(self.conv1(input))))
        x = F.leaky_relu((self.conv2(x)))      #self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.leaky_relu((self.conv3(x)))      #self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.leaky_relu((self.conv4(x)))      #self.pool4(F.relu(self.bn4(self.conv4(x))))
        #
        # x = F.leaky_relu(self.bn1(self.conv1(input)))  #self.pool1(F.relu(self.bn1(self.conv1(input))))
        # x = F.leaky_relu(self.bn2(self.conv2(x)))      #self.pool2(F.relu(self.bn2(self.conv2(x))))
        # x = F.leaky_relu(self.bn3(self.conv3(x)))      #self.pool3(F.relu(self.bn3(self.conv3(x))))
        # x = F.leaky_relu(self.bn4(self.conv4(x)))      #self.pool4(F.relu(self.bn4(self.conv4(x))))
        # x = F.leaky_relu(self.bn5(self.conv5(x)))
        # x = F.leaky_relu(self.bn6(self.conv6(x)))
        # x = F.leaky_relu(self.bn7(self.conv7(x)))
        # x = F.leaky_relu(self.bn8(self.conv8(x)))

        # x = F.leaky_relu((self.conv1(input)))  #self.pool1(F.relu(self.bn1(self.conv1(input))))
        # x = F.leaky_relu((self.conv2(x)),)      #self.pool2(F.relu(self.bn2(self.conv2(x))))
        # x = F.leaky_relu((self.conv3(x)))      #self.pool3(F.relu(self.bn3(self.conv3(x))))
        # x = F.leaky_relu((self.conv4(x)))      #self.pool4(F.relu(self.bn4(self.conv4(x))))
        # x = F.leaky_relu((self.conv5(x)))
        # x = F.leaky_relu((self.conv6(x)))
        # x = F.leaky_relu((self.conv7(x)))
        # x = F.leaky_relu(self.bn8(self.conv8(x)))




        # x = self.pool1(F.relu(self.bn1(self.conv1(input))))
        # x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        # x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        # x = self.pool6(F.relu(self.bn6(self.conv6(x))))

        # x = self.pool7(F.relu(self.bn7(self.conv7(x))))
        # x = self.adaptpool(x)



        x = x.view(x.size(0),-1)

        x = (self.fc1(x))
        # x = x.view(x.size(0), self.height,self.width)
        x = x.view(x.size(0), self.height*self.width)
        # x = self.fc2(x)
        #
        # x = self.fc3(x)
        # x = self.fc4(x)
        return x

if __name__ == '__main__':

    pass
