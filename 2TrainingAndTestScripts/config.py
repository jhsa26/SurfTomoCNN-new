#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
Author: Jing Hu
Date: 2020-11-23 14:36:51
LastEditTime: 2020-11-23 20:57:17
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /SurfTomoCNN-new/config.py
'''
class Config(object):
    def __init__(self):
        self.filepath_disp_training = '../DataSet/TrainingData/0.5km/USA_Tibet/disp_combine_gaussian_map/'
        self.filepath_vs_training   = '../DataSet/TrainingData/0.5km/USA_Tibet/vs_curve/'

        self.filepath_disp_real    = '../DataSet/TestData/real-8s-50s/China/disp_pg_real/' 
        self.batch_size = 64     # training batch size
        self.nEpochs = 600       # maximum number of epochs to train for
        self.lr = 0.00001        # learning rate
        self.seed = 123          # random seed to use. Default=123
        self.plot = True         # show validation result during training
        self.alpha=0.0000        # damping, not used here 
        self.testsize=0.2
        self.pretrained =True
        self.start=600
        self.pretrain_net = "./model_para/model_epoch_"+str(self.start)+".pth"
if __name__ == '__main__':
    pass
