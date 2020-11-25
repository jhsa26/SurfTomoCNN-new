'''
Author: your name
Date: 2020-11-23 14:51:36
LastEditTime: 2020-11-23 20:04:48
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /SurfTomoCNN-new/util.py
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @Author: HUJING
 @Time:5/14/18 10:11 AM 2018
 @Email: jhsa26@mail.ustc.edu.cn
 @Site:jhsa26.github.io
 """

import os
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config
import random
import torch
import torch.nn as nn

def get_gaussian_map(vel,vel_axis,radius=0.1):
    rows  = vel_axis.shape[0]
    cols  = vel.shape[0]
    vel_map = np.zeros((rows,cols))
    for i in range(cols):
        vel_temp = vel[i]
        x_gaussian = _gaussian(vel_temp, vel_axis, r=radius)
        vel_map[:, i] = x_gaussian
    return vel_map

def _gaussian(vel,vel_axis,r=0.1):
    x_gaussian = np.exp(-((vel_axis-vel)**2)/(2*r**2))
    return x_gaussian

def randomFix(seed=9999):
    # fixed seed, because pytoch initilize weight randomly
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        print("cuda is available")
    else:
        print("cuda is not available")
        torch.manual_seed(seed)

def checkpoint(epoch, model):
    torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % ('./model_para', epoch))
    print("Checkpoint saved to {}".format('%s/model_epoch_%d.pth' % ('./model_para', epoch)))

def weights_init(m):
    classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.xavier_normal_(m.weight.data)
        # nn.init.orthogonal_(m.weight.data)
        # m.weight.data.fill_(0)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.xavier_normal_(m.weight.data)
        # nn.init.orthogonal_(m.weight.data)
        # m.weight.data.fill_(0)
        m.bias.data.fill_(0)
"""
def loss_fn(alpha, model, y_pred, y_obs, crit=nn.MSELoss()):
    # l2_reg = torch.tensor(1.,requires_grad=True)
    # for param in model.parameters():
    #     l2_reg=l2_reg+torch.norm(param)
    # # loss = crit(y_pred, y_obs)
    loss = crit(y_pred, y_obs)  # +alpha*l2_reg
    return loss
""" 
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    def forward(self, pred, truth):
        # note np.gradient
        pred_grad = pred[:, 1:-1] - pred[:, 0:-2]
        truth_grad = truth[:, 1:-1] - truth[:, 0:-2]
      #  a = (pred_grad - truth_grad) #changed 08052019
        c = (pred - truth)
        d = torch.norm(c) / np.sqrt(torch.numel(c)) #+ torch.norm(a) / np.sqrt(torch.numel(a))  #change 08052019
        return d
#def count_parameters(model):

#    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# obtain weights
def extract_weights(model):
    # for k,v in model.state_dict().iteritems():
    # print("Layer {} ".format(k))
    # print(v)
    norm_weight = []
    a = 0
    for layer in model.modules():
        a = a + 1
        if isinstance(layer, nn.Linear):
            norm_weight.append(torch.norm(layer.weight, 2))
            # print("Layer {} linear  ".format(a))
        if isinstance(layer, nn.Conv2d):
            norm_weight.append(torch.norm(layer.weight, 2))
            # print("Layer {} conv2  ".format(a))
    return norm_weight

# def  writerlogfile(writer,norm_weight,epoch, tloss, vloss,vrms, vloss_hist):

def  writerlogfile(writer,epoch, tloss, vloss,vrms):
    '''
    for tensorboardX to show training and validation loss evolution
    '''
    # writer.add_scalar('net/conv1', norm_weight[0], epoch)
    # writer.add_scalar('net/conv2', norm_weight[1], epoch)
    # writer.add_scalar('net/conv3', norm_weight[2], epoch)
    # writer.add_scalar('net/conv4', norm_weight[3], epoch)
    # writer.add_scalar('net/Linear1', norm_weight[4], epoch)
    # writer.add_scalar('net/conv5', norm_weight[4], epoch)
    # writer.add_scalar('net/conv6', norm_weight[5], epoch)
    # writer.add_scalar('net/conv7', norm_weight[6], epoch)
    # writer.add_scalar('net/conv8', norm_weight[7], epoch)
    # writer.add_scalar('net/Linear1', norm_weight[8], epoch)
    # writer.add_scalar('net/Linear2', norm_weight[9], epoch)
    writer.add_scalar('data/TrainingLoss', tloss, epoch)
    writer.add_scalar('data/ValidationLoss', vloss, epoch)
    writer.add_scalar('data/ValidationTrueRMS', vrms, epoch)
    # writer.add_histogram('data/ValidationLossHistorgram', vloss_hist, global_step=epoch, bins='tensorflow')
    return writer
if __name__ == '__main__':
    pass
