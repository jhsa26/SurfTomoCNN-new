#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@Time   15/05/2018 11:41 AM 2018
@Author HJ@USTC
@Email  jhsa26@mail.ustc.edu.cn
@Blog   jhsa26.github.io
"""

class Config(object):
    def __init__(self):

#        self.filepath_disp_training = '../DataSet/TrainingData/0.5km/China/disp_combine_gaussian_map_5-10/'
        self.filepath_disp_training = '../DataSet/TrainingData/0.5km/USA/disp_combine_gaussian_map/'
        #self.filepath_vs_training   = '../DataSet/TrainingData/0.5km/China/vs_curve_5-10/'
        self.filepath_vs_training   = '../DataSet/TrainingData/0.5km/USA/vs_curve/'
#        self.filepath_disp_pred     = '../../DataSet/TestData/real-8s-50s/USA/disp_combine_gaussian_map_utah/'
        self.filepath_vs_pred       = '../DataSet/TestData/real-8-50s/vs_syn_gaussian/'
        self.filepath_disp_pred     = '../DataSet/TestData/real-8s-50s/China/disp_combine_gaussian_map_test_4512/'
        self.batch_size = 64     # training batch size
        self.nEpochs = 300          # umber of epochs to train for
        self.lr = 0.000005
        self.seed = 123             # random seed to use. Default=123
        self.plot = True
        self.alpha=0.0000             # damping
        self.testsize=0.2
        self.pretrained =True
        self.start=600
        self.pretrain_net = "./model_para_USA/model_epoch_"+str(self.start)+".pth"
if __name__ == '__main__':
    pass
