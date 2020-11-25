#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : Mian_Predict.py
@Description :
@Time    : 2020/11/23 20:02:03
@Author  : Jing Hu
@Email   : jhsa920@163.com
@Version : 1.0
'''
import torch
import torch.nn.functional as F
from src.util import get_gaussian_map,randomFix
from src.NetModel_curve import Net as Net
from config import Config
from scipy.signal import savgol_filter as smooth
import numpy as np
import os
from src.loadData import readRealData
os.system("rm -rf vs_out && mkdir vs_out" )
vel_axis = np.linspace(2,5,num=60)
radius=0.1 # fixted, same parameter used to generate training dataset
num_perturbation=1 # used for uncertainty analysis via bootstrapping test. if num_perturbation is 1, not do bootstrapping test
option = Config()
# fixed seed, because pytoch initilize weight randomly
randomFix(option.seed)
model = Net(image_width=17,
             image_height=60,
             image_outwidth=301,
             image_outheight=1,
             inchannel=2,outchannel=4)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu"    if torch.cuda.is_available() else "cpu")
model.to(device)
if option.pretrained:
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(option.pretrain_net))
    else:
        model.load_state_dict(torch.load(option.pretrain_net,map_location={'cuda:0':'cpu'}))
else:
    print('no pretrained model')
    os._exit(0)
print('===> Predicting')
model.cpu()
epoch=option.start
r=radius

model.eval()
disp_pg_real,locKeys=readRealData(option.filepath_disp_real)
vel_pred_total=[]
depth=np.arange(0,150.5,0.5); depth=depth[:,np.newaxis]
num_disp=disp_pg_real.shape[0]
for i in range(num_disp):    
    vel_count=[]
    # dispersion axis
    temp= disp_pg_real[i,:,:]
    phase=temp[:,1];phase_un=temp[:,2]
    group=temp[:,3];group_un=temp[:,4]
    #convert map
    for j in range(num_perturbation):
        new_phase=phase+phase_un*(np.random.rand(17)-0.5)*2
        new_group=group+group_un*(np.random.rand(17)-0.5)*2
        if j==0:
            new_phase=phase[:]
            new_group=group[:]
        vel_map_p=get_gaussian_map(new_phase,vel_axis,radius=r) 
        vel_map_g=get_gaussian_map(new_group,vel_axis,radius=r)
        input=torch.Tensor([vel_map_p,vel_map_g])
        input = input.view([1,input.size(0),input.size(1),input.size(2)])
        # compute output
        output = model(input)  # output[batchsize,H,W]
        output = output.view([output.size(1)]).data.numpy()
        vel_pred = smooth(output,5,3)
        vel_count.append(vel_pred)
    vel=np.array(vel_count).T
    vel_loc= "{:.2f}_{:.2f}".format(locKeys[i,0],locKeys[i,1]) # 
    print(str(i)+"  "+vel_loc)
    prefixname = vel_loc
    mean=np.expand_dims(np.mean(vel,axis=1),axis=1)
    std=np.expand_dims(np.std(vel,axis=1),axis=1)
    vel1=np.expand_dims(vel[:,0],axis=1)
    #print(vel.shape,depth.shape,np.std(vel,axis=1)) 
    output=np.hstack((depth,mean,std,vel))
    np.savetxt("./vs_out/"+prefixname+'.txt',output,fmt='%10.5f')
