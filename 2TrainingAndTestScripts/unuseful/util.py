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
from torch.autograd import Variable
class Reader_test(object):
    def __init__(self):
        self.config = Config()
        self.filepath_disp_pred = self.config.filepath_disp_pred
        self.disp_filenames_pred, self.keys_pred = self.get_realdata_filename()
    def get_realdata_filename(self):
        filename_dispersion_total = []
        key_total = []
        if os.path.exists(self.filepath_disp_pred):
            files_disp = os.listdir(self.filepath_disp_pred)
            # read inputs
            for file in files_disp:
                key = file[0:-4]
                filename_disp = self.filepath_disp_pred + file
                if os.path.exists(filename_disp):
                    filename_dispersion_total.append(filename_disp)
                    key_total.append(key)
            return np.array(filename_dispersion_total), np.array(key_total)
        else:

            print('Input test file path is not exist, check the input path!')
            return None, None
    def read_curves(self, filenames, data_type):
        data = []
        if data_type == 'disp':
            for file in filenames:
                temp_data_temp = np.loadtxt(file)
                data.append(temp_data_temp)
            return np.array(data)
        else:
            print('check the data_type, which must be "vs" or "disp" ')
            return None

    def get_real_disp_curves(self):
        disp_filenames = self.disp_filenames_pred
        keys = self.keys_pred
        filenames = disp_filenames  
        test_x = self.read_curves(filenames, 'disp')
        test_keys = keys
        return test_x, test_keys


















class Reader(object):
    def __init__(self):
        self.config = Config()
        self.filepath_disp_training = self.config.filepath_disp_training
        self.filepath_vs_training = self.config.filepath_vs_training
        self.filepath_disp_pred = self.config.filepath_disp_pred
        self.filepath_vs_pred = self.config.filepath_vs_pred
        self.batch_size = self.config.batch_size
        self.disp_filenames, self.vs_filenames, self.keys = self.get_train_filename()
        self.disp_filenames_pred, self.keys_pred = self.get_realdata_filename()

    def get_batch_file(self):
        disp_filenames = self.disp_filenames
        vs_filenames = self.vs_filenames
        keys = self.keys

        train_pos, test_pos = train_test_split(range(len(keys)), test_size=Config().testsize, random_state=42)
        random.shuffle(train_pos)
        random.shuffle(test_pos)

        train_keys_num = len(train_pos)

        batch_size = self.batch_size
        batch_num = int(float(train_keys_num) / float(batch_size)) + 1
        batch_array = []
        for i in range(1, batch_num + 1):
            index1 = i * batch_size
            batch_array.append(index1 - 1)
        if batch_array[-1] >= len(train_pos) - 1:
            batch_array[-1] = len(train_pos) - 1
        return batch_array, train_pos, test_pos

    def get_train_filename(self):
        filename_dispersion_total = []
        filename_vs_total = []
        key_total = []
        if os.path.exists(self.filepath_disp_training) and os.path.exists(self.filepath_vs_training):
            files_disp = os.listdir(self.filepath_disp_training)
            # read inputs
            for file in files_disp:

                key = file[2:-4]  # add group disp and phase disp

                filename_disp = self.filepath_disp_training + file

                filename_vs = self.filepath_vs_training + file
                if os.path.exists(filename_vs) and os.path.exists(filename_disp):
                    filename_dispersion_total.append(filename_disp)

                    filename_vs_total.append(filename_vs)

                    key_total.append(key)

            return np.array(filename_dispersion_total), np.array(filename_vs_total), np.array(key_total)
        else:
            print('Input train file path is not exist, check the input path!')
            return None, None, None

    def get_predsyn_filename(self):
        filename_dispersion_total = []
        filename_vs_total = []
        key_total = []
        if os.path.exists(self.filepath_disp_pred):
            files_disp = os.listdir(self.filepath_disp_pred)
            # read inputs
            for file in files_disp:
                key = file[2:-4]
                filename_disp = self.filepath_disp_pred + file
                filename_vs = self.filepath_vs_pred + file
                if os.path.exists(filename_disp) and os.path.exists(filename_vs):
                    filename_dispersion_total.append(filename_disp)
                    filename_vs_total.append(filename_vs)
                    key_total.append(key)
            return np.array(filename_dispersion_total), np.array(filename_vs_total), np.array(key_total)
        else:
            print('Input test file path is not exist, check the input path!')
            return None, None, None

    def get_realdata_filename(self):
        filename_dispersion_total = []
        key_total = []
        if os.path.exists(self.filepath_disp_pred):

            files_disp = os.listdir(self.filepath_disp_pred)
            # read inputs
            for file in files_disp:
                key = file[2:-4]
                filename_disp = self.filepath_disp_pred + file
                if os.path.exists(filename_disp):
                    filename_dispersion_total.append(filename_disp)
                    key_total.append(key)
            return np.array(filename_dispersion_total), np.array(key_total)
        else:

            print('Input test file path is not exist, check the input path!')
            return None, None

    def get_batch_data(self, data_type, index1, index2, train_pos, test_pos):
        disp_filenames = self.disp_filenames
        vs_filenames = self.vs_filenames
        keys = self.keys

        if data_type == 'train':
            filenames = disp_filenames[train_pos[index1:index2 + 1]]
            batch_x = self.readdata(filenames, 'disp')
            filenames = vs_filenames[train_pos[index1:index2 + 1]]
            batch_y = self.readdata(filenames, 'vs')
            batch_keys = keys[train_pos[index1:index2 + 1]]
            return batch_x, batch_y, batch_keys

        elif data_type == 'test':
            sample_indexs = random.sample(test_pos, len(test_pos))
            filenames = disp_filenames[sample_indexs];
            test_x = self.readdata(filenames, 'disp')
            filenames = vs_filenames[sample_indexs];
            test_y = self.readdata(filenames, 'vs')
            test_keys = keys[test_pos]
            return test_x, test_y, test_keys
        else:
            print('check the data_type, which must be "train" or "test" ')
            return None, None, None

    def readdata(self, filenames, data_type):
        data = []
        if data_type == 'disp':
            for file in filenames:
                temp_data = []
                with open(file, 'r') as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        temp = list(map(float, line.split()))
                        temp_data.append(temp)

                temp_data_temp = np.array(temp_data)
                # according to your activation function range
                #
                # remean_temp = temp_data_temp[:, 1] - np.mean(temp_data_temp[:, 1])
                # temp_data_temp[:, 1] = (remean_temp[:]-np.min(remean_temp[:]))/(np.max(remean_temp)-np.min(remean_temp)+0.01)

                data.append(temp_data_temp[:, 1:3])
            return np.array(data)
        elif data_type == "vs":
            for file in filenames:
                temp_data = []
                with open(file, 'r') as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        temp = list(map(float, line.split()))
                        temp_data.append(temp)
                temp_data_temp = np.array(temp_data)
                data.append(temp_data_temp[0:300, 0:2])
            return np.array(data)
        else:
            print('check the data_type, which must be "vs" or "disp" ')
            return None

    def get_batch_gaussian_map(self, data_type, index1, index2, train_pos, test_pos):
        disp_filenames = self.disp_filenames
        vs_filenames = self.vs_filenames
        keys = self.keys

        if data_type == 'train':
            filenames = disp_filenames[train_pos[index1:index2 + 1]]
            batch_x = self.read_gaussian_map(filenames, 'disp')
            filenames = vs_filenames[train_pos[index1:index2 + 1]]
            batch_y = self.read_gaussian_map(filenames, 'vs')
            batch_keys = keys[train_pos[index1:index2 + 1]]
            return batch_x, batch_y, batch_keys

        elif data_type == 'test':
            sample_indexs = random.sample(test_pos, len(test_pos))
            filenames = disp_filenames[sample_indexs];
            test_x = self.read_gaussian_map(filenames, 'disp')
            filenames = vs_filenames[sample_indexs]
            test_y = self.read_gaussian_map(filenames, 'vs')
            test_keys = keys[test_pos]
            return test_x, test_y, test_keys
        else:
            print('check the data_type, which must be "train" or "test" ')
            return None, None, None

    def read_gaussian_map(self, filenames, data_type):
        data = []
        if data_type == 'disp':
            for file in filenames:
                temp_data_temp = np.load(file)
                data.append(temp_data_temp)
            return np.array(data)
        elif data_type == "vs":
            for file in filenames:
                temp_data_temp = np.load(file)
                data.append(temp_data_temp)
            return np.array(data)
        else:
            print('check the data_type, which must be "vs" or "disp" ')
            return None

    def get_gaussian_map_predsyn(self):
        disp_filenames = self.disp_filenames_pred
        vs_filenames = self.vs_filenames_pred
        keys = self.keys_pred
        # sample_indexs = random.sample(range(len(keys)), len(keys))
        filenames = disp_filenames  # [sample_indexs];
        test_x = self.read_gaussian_map(filenames, 'disp')
        filenames = vs_filenames  # [sample_indexs]
        test_y = self.read_gaussian_map(filenames, 'vs')
        test_keys = keys
        return test_x, test_y, test_keys

    def get_real_gaussian_map(self):
        disp_filenames = self.disp_filenames_pred
        keys = self.keys_pred
        filenames = disp_filenames  # [sample_indexs];
        # print(filenames)
        test_x = self.read_gaussian_map(filenames, 'disp')
        test_keys = keys
        return test_x, test_keys

    def get_batch_disp_gaussian_map_vs_curve(self, data_type, index1, index2, train_pos, test_pos):
        disp_filenames = self.disp_filenames
        vs_filenames = self.vs_filenames
        keys = self.keys

        if data_type == 'train':
            filenames = disp_filenames[train_pos[index1:index2 + 1]]
            batch_x = self.read_gaussian_map(filenames, 'disp')
            filenames = vs_filenames[train_pos[index1:index2 + 1]]
            batch_y = self.read_vs_curves(filenames, 'vs')
            batch_keys = keys[train_pos[index1:index2 + 1]]
            return batch_x, batch_y, batch_keys

        elif data_type == 'test':
            sample_indexs = random.sample(test_pos, len(test_pos))
            filenames = disp_filenames[sample_indexs];
            test_x = self.read_gaussian_map(filenames, 'disp')
            filenames = vs_filenames[sample_indexs]
            test_y = self.read_vs_curves(filenames, 'vs')
            test_keys = keys[test_pos]
            return test_x, test_y, test_keys
        else:
            print('check the data_type, which must be "train" or "test" ')
            return None, None, None

    def read_vs_curves(self, filenames, data_type):
        data = []
        for file in filenames:
            temp_data_temp = np.loadtxt(file)
            data.append(temp_data_temp[:, 1])
        return np.array(data)


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


def loss_fn(alpha, model, y_pred, y_obs, crit=nn.MSELoss()):
    # l2_reg = torch.tensor(1.,requires_grad=True)
    # for param in model.parameters():
    #     l2_reg=l2_reg+torch.norm(param)
    # # loss = crit(y_pred, y_obs)
    loss = crit(y_pred, y_obs)  # +alpha*l2_reg
    return loss


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        print(1)

    def forward(self, pred, truth):
        # note np.gradient
        pred_grad = pred[:, 1:-1] - pred[:, 0:-2]
        truth_grad = truth[:, 1:-1] - truth[:, 0:-2]
      #  a = (pred_grad - truth_grad) #changed 08052019
        c = (pred - truth)
        d = torch.norm(c) / np.sqrt(torch.numel(c)) #+ torch.norm(a) / np.sqrt(torch.numel(a))  #change 08052019
        return d


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


## simple stepping rates


class StepLR(object):
    def __init__(self, pairs):
        super(StepLR, self).__init__()
        N = len(pairs)
        rates = []
        steps = []
        for n in range(N):
            s, r = pairs[n]
            if r < 0: s = s + 1
            steps.append(s)
            rates.append(r)
        self.rates = rates
        self.steps = steps

    def get_rate(self, epoch=None):
        N = len(self.steps)
        lr = -1
        for n in range(N):
            if epoch >= self.steps[n]:
                lr = self.rates[n]
        return lr

    def __str__(self):
        string = 'Step Learning Rates\n' \
                 + 'rates=' + str(['%7.4f' % i for i in self.rates]) + '\n' \
                 + 'steps=' + str(['%7.0f' % i for i in self.steps]) + ''
        return string


## https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py
class DecayLR(object):
    def __init__(self, base_lr, decay, step):
        super(DecayLR, self).__init__()
        self.step = step
        self.decay = decay
        self.base_lr = base_lr

    def get_rate(self, epoch=None, num_epoches=None):
        lr = self.base_lr * (self.decay ** (epoch // self.step))
        return lr

    def __str__(self):
        string = '(Exp) Decay Learning Rates\n' \
                 + 'base_lr=%0.3f, decay=%0.3f, step=%0.3f' % (self.base_lr, self.decay, self.step)
        return string


# 'Cyclical Learning Rates for Training Neural Networks'- Leslie N. Smith, arxiv 2017
#       https://arxiv.org/abs/1506.01186
#       https://github.com/bckenstler/CLR

class CyclicLR(object):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step = step
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: (0.5) ** (x - 1)
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step != None:
            self.step = new_step
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step))
        x = np.abs(self.clr_iterations / self.step - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def get_rate(self, epoch=None, num_epoches=None):
        self.trn_iterations += 1
        self.clr_iterations += 1

        lr = self.clr()

        return lr

    def __str__(self):
        string = 'Cyclical Learning Rates\n' \
                 + 'base_lr=%0.3f, max_lr=%0.3f' % (self.base_lr, self.max_lr)
        return string


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def adjust_learning_rate(init_lr, optimizer, epoch, decay_rate=0.5):
    lr = init_lr * (decay_rate ** (epoch // 10))
    print(lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


if __name__ == '__main__':
    pass
