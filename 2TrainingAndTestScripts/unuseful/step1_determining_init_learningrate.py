#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
 @Author: HUJING
 @Time:5/21/18 3:36 PM 2018
 @Email: jhsa26@mail.ustc.edu.cn
 @Site:jhsa26.github.io
 """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util import *
from NetModel import Net as Net
from config import Config
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import pickle
import time
from Main_plotResults import Mainplot
writer = SummaryWriter()
option = Config()
alpha=option.alpha
plt.ion()
# fixed seed, because pytoch initilize weight randomly
torch.manual_seed(option.seed)

print('===> Loading datasets')
batch_index, train_pos, test_pos = Reader().get_batch_file()

print('===> Building net')
model = Net(inchannel=2,outchannel=8)
model.apply(weights_init)

optimizer = optim.Adam(model.parameters(),  lr=option.lr,weight_decay=alpha)
option.lr= 1e-6
print('===> Training Net')



def training(epoch):
    epoch_loss =[]
    accuracy = []
    total_num = len(train_pos)
    lr_mult = (1 / 1e-5) ** (1 / 100)
    lr = [option.lr]
    losses = []
    best_loss = 1e9
    for iteration in range(len(batch_index)):

        index1 = iteration*option.batch_size
        index2 = batch_index[iteration]
        batch_x,batch_y,batch_loc= Reader().get_batch_data('train',index1, index2,train_pos,test_pos)
        inputs = torch.Tensor(batch_x[:,:,:])
        #  batch_size,channels,H,W
        inputs = inputs.view([inputs.size(0),inputs.size(2),1,inputs.size(1)])
        targets = batch_y[:,:,1]
        targets = torch.Tensor(targets)
        optimizer.zero_grad()
        outputs = model(inputs)
        # cost function
        loss = loss_fn(alpha,model,outputs, targets)
        loss.backward()
        optimizer.step()
        # print statistics
        epoch_loss.append(loss.item())
        if iteration%20==19:
            num = index2
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, num, total_num,
                                                                           100. * num/total_num, loss.item()))




        outvspattern = Vel_pattern().get_Vel_pattern_array(outputs)
        targetsvspattern = Vel_pattern().get_Vel_pattern_array(targets)
        temppatterns = (targetsvspattern - outvspattern)
        accuracy.append((temppatterns.nelement()-len(torch.nonzero(temppatterns)))/temppatterns.nelement())
        if loss.item() < best_loss:
            best_loss = loss.item()
        if loss.item() > 10* best_loss or lr[-1]> 1.0:
             break
        # if  lr[-1]> 1.0:
        #     break

        for g in optimizer.param_groups:
            g['lr'] = g['lr']*lr_mult
        lr.append(g['lr']*lr_mult)

    lr = np.array(lr)
    losses = np.array(losses)
    accuracy=np.array(accuracy)


    return lr,accuracy,epoch_loss


for epoch in range(1, 2):

    # adjust_learning_rate(optimizer, epoch,decay_rate=0.5)
    model.train()
    lr, accuracy, epoch_loss= training(epoch)
    model.eval()
    plt.subplot(3,1,1)
    plt.xticks(np.log([1e-6,1e-5 ,1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-6,1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.plot(np.log(lr), epoch_loss)
    plt.subplot(3,1,2)
    plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
    plt.xlabel('learning rate')
    plt.ylabel('accuracy')
    plt.plot(np.log(lr), accuracy)
    plt.subplot(3, 1, 3)
    plt.xlabel('num iterations')
    plt.ylabel('learning rate')
    plt.plot(lr)
    plt.tight_layout()
    plt.pause(100)
    plt.show()

