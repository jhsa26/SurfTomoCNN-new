#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : Main_Train.py
@Description :
@Time    : 2020/11/23 14:50:36
@Author  : Jing Hu
@Email   : jhsa920@163.com
@Version : 1.0
'''

import matplotlib as mpl
# mpl.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import numpy as np
import time
import sys
sys.path.append('./src')
from src.NetModel_curve import Net as Net
from config import Config
from torch.utils.data import DataLoader
from src.loadData import ToTensor, DispVsDataset,getTrainValidationFiles
from torchsummary import summary
from src.util import randomFix,checkpoint,writerlogfile,MyLoss,weights_init
import os
def training(epoch, model,optimizer,option,trainDataset,device,loss_fn=nn.MSELoss()):
    # adjust learning rate
    # optimizer = adjust_learning_rate(option.lr,optimizer, epoch)
    trainLoader = DataLoader(trainDataset, batch_size=option.batch_size,shuffle=True, num_workers=0)
    total_num=len(trainDataset) 
    num=0
    epoch_loss=0.0
    current_loss=None
    for batch_i,sample_batch in enumerate(trainLoader):
        inputs,targets=sample_batch['disp'].to(device),sample_batch['vs'].to(device) # cnn input and labels
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        # print statistics
        num = len(inputs) + num
        current_loss = loss.item()
        epoch_loss += current_loss
        if batch_i % 20 == 19:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, num, total_num,
                                                                           100. * num / total_num, loss.item()))
    total_batch=len(trainLoader)
    average_loss = epoch_loss / total_batch
    print("===> Epoch {} Complete: Avg. Loss: {:.4f} ".format(epoch, average_loss))
    # return the last iteration loss.
    return current_loss  

def test(epoch, model, option,validDataset,loss_fn=nn.MSELoss()):
    if option.plot:
        fig = plt.figure(num=1, figsize=(12, 8), dpi=80, clear=True)
    validLoader=DataLoader(validDataset,shuffle=True,batch_size=option.batch_size,num_workers=0)
    test_loss = 0
    rms=0
    model.cpu()
    model.eval() #
    for batch_i, sample_batch in enumerate(validLoader):
        # dispersion
        input = sample_batch['disp']# input = input.view([1, input.size(0), input.size(1), input.size(2)])
        # velocity 
        label = sample_batch['vs'] #         label = label.view([1, label.size(0)])
        locationKey=sample_batch['location'].numpy()
        # compute output
        output = model(input)  # output[batchsize,H,W]
        loss = loss_fn(output, label).item() 
        test_loss += loss # sum up batch loss
        # collect output loss, need to delete
        vel_pred = output.detach().numpy()
        vel_syn =  label.numpy()
        res = (vel_pred - vel_syn)
        rms = rms + np.sqrt(np.power(res, 2).sum() / res.size)
        if option.plot:
            if batch_i % 4 is 0:
                select_one=np.random.randint(0,option.batch_size)
                vel_syn=vel_syn[select_one,:];vel_pred=vel_pred[select_one,:]
                plt.plot(vel_syn,  np.arange(0, len(vel_syn),1)*0.5, '-.', color='red')
                plt.plot(vel_pred, np.arange(0, len(vel_pred),1)*0.5, '-', color='green')
                plt.title('True')
                plt.xlabel('Vs(km/s)')
                plt.ylabel('Depth(km)')
                plt.gca().invert_yaxis()
                plt.savefig('./Figs/Fitting_epoch{:.0f}_{:.3f}_{:.3f}.jpg'.format(epoch,locationKey[select_one,0],locationKey[select_one,1]), dpi=300)
                plt.pause(0.01)
                fig.clear()
                pass
    total_batch=len(validLoader)
    average_loss = test_loss /total_batch
    rms = rms/total_batch
    print("===> Avg. test loss: {:.4f} {:.4f}  ".format(average_loss, rms))
    return average_loss,rms 


def main():
    os.system("test -d output|| mkdir output")
    os.system("test -d Figs|| mkdir Figs")
    os.system("test -d model_para|| mkdir model_para")
    writer = SummaryWriter()
    option = Config()
    alpha = option.alpha
    randomFix(option.seed); # fix seed,reproduce our results for each run-time 

    print('===> Building net')
    model = Net(image_width=17,
                image_height=60,
                image_outwidth=301,
                image_outheight=1,
                inchannel=2, outchannel=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # for one gpu
    model.to(device)         # assign a gpu or cpu 
    summary(model,(2,17,60)) # how many parameters of designed network
    print(model)
    # initialize weights of networks
    if option.pretrained:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(option.pretrain_net))
        else:
            model.load_state_dict(torch.load(option.pretrain_net,map_location={'cuda:0':'cpu'}))
    else:
        model.apply(weights_init)
    if option.plot:
        plt.ion()
    # set optimizer
    # optimizer = optim.Adam(model.parameters(), lr=option.lr, weight_decay=alpha)
    optimizer = optim.RMSprop(model.parameters(), lr=option.lr, weight_decay=alpha)
    # optimizer = optim.SGD(model.parameters(),  lr=option.lr,weight_decay=alpha,momentum=0.9)
    

    print('===> load train and validation Dataset')
    dispPath=option.filepath_disp_training
    vsPath=option.filepath_vs_training
    trainFiles,validFiles=getTrainValidationFiles(dispPath,vsPath,validSize=0.2)
    trainDataset=DispVsDataset(trainFiles,transform=ToTensor())
    validDataset=DispVsDataset(validFiles,transform=ToTensor())
    print('===> Training Net')
    time_loc = list(map(str, list(time.localtime())))
    time_loc = '_'.join(time_loc[0:-5])
    with  open('output/epochInfo' + time_loc + '.txt', 'w') as f:
        for epoch in range(option.start, option.start + option.nEpochs + 1):
            # training 
            tloss = training(epoch, model,optimizer,option,trainDataset,device,loss_fn=MyLoss())
            # # validation
            vloss, vrms = test(epoch, model,option,validDataset,loss_fn=MyLoss())
            model.to(device)
            # write log file
            writer = writerlogfile(writer, epoch, tloss, vloss, vrms)
            if epoch % 20 is 0:
                checkpoint(epoch, model)
            elif epoch == 1:
                checkpoint(epoch, model)
            string_output = "{}".format("%d %10.7f %10.7f %10.7f %s" % (epoch, tloss, vloss, vrms, '\n'))
            f.write(string_output)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    checkpoint(epoch, model)
    print('Finished Training')
if __name__ == '__main__':
    main()
    pass
