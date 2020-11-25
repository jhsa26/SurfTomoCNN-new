'''
Author: your name
Date: 2020-11-23 18:58:39
LastEditTime: 2020-11-23 19:55:07
LastEditors: your name
Description: In User Settings Edit
FilePath: /tmpp/unuseful/Main_plotResults.py
'''

# coding: utf-8

# In[18]:

#!/usr/bin/env python3  

"""  
 @Author: HUJING
 @Time:5/29/18 10:44 PM 2018
 @Email: jhsa26@mail.ustc.edu.cn
 @Site:jhsa26.github.io
 """
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
from mpl_toolkits.axes_grid.inset_locator import inset_axes
os.system('test -d  Figs_show || mkdir Figs_show')
time_loc='2018_12_25_21'
time_loc='2020_11_23_19'
epoch=60
fontsize = 22
linewidth = 1.5
temp = np.loadtxt('../output/epochInfo'+time_loc+'.txt')
fig = plt.figure(num=1, figsize=(8, 6), dpi=80, clear=True)
epoch_train_collection    = temp[0:epoch,0]
ave_train_loss_collection = temp[0:epoch,1]
ave_test_loss_collection  = temp[0:epoch,2]
ave_test_rms_collection   = temp[0:epoch,3]
# plot epoch-training loss and epoch-validation loss
plt.plot(epoch_train_collection,ave_train_loss_collection,'-bo',linewidth=2,markersize=5,
         label='Training loss')
plt.plot(epoch_train_collection,ave_test_loss_collection,'-ro',linewidth=2,markersize=5,
         label= 'Validation loss')
# plt.plot(epoch_train_collection, ave_test_rms_collection, '-go', linewidth=2, markersize=5,
#          label='validation true loss')
plt.legend(loc=0,fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('Epoch',fontsize=fontsize)
plt.ylabel('Loss',fontsize=fontsize)
plt.title('Epoch-Loss Curve',fontsize=fontsize)
ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(linewidth)
ax.tick_params(axis='y', direction='in', length=6, labelrotation=0, width=2, bottom=True, top=False, left=True,
               right=False)
ax.tick_params(axis='x', direction='in', length=6, labelrotation=0, width=2, bottom=True, top=False, left=True,
               right=False)




inset_axes = inset_axes(ax, 
                    width="50%", # width = 30% of parent_bbox
                    height="40%", # height : 1 inch
                    loc='center' )


epoch_train_collection    = temp[20:epoch,0]
ave_train_loss_collection = temp[20:epoch,1]
ave_test_loss_collection  = temp[20:epoch,2]
ave_test_rms_collection   = temp[20:epoch,3]

plt.plot(epoch_train_collection,ave_train_loss_collection,'-bo',linewidth=2,markersize=5,
         label='training loss')
plt.plot(epoch_train_collection,ave_test_loss_collection,'-ro',linewidth=2,markersize=5,
         label= 'validation loss')
ax = plt.gca() 
ax.tick_params(axis='both', direction='in', length=6, labelrotation=0, width=2, bottom=True, top=False, left=True,
               right=False,labelsize=fontsize-6) 
 



name = '../Figs/EpochLossCurve_'+time_loc+'epoch_'+str(epoch)+".png"
plt.savefig(name, dpi=300,bbox_inches="tight")
plt.pause(0.5)
fig.clear()

