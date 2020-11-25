#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
 @Author: HUJING
 @Time:5/21/18 3:36 PM 2018
 @Email: jhsa26@mail.ustc.edu.cn
 @Site:jhsa26.github.io
 """

import matplotlib as mpl

# mpl.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import pickle
import time
# import sys
# sys.path.append('./src')
from src.util import *
from src.writerlogfile import writerlogfile
from src.NetModel_curve import Net as Net
from config import Config
from torchsummary import summary
myloss = MyLoss()



Input_fun=Reader()
#def count_parameters(model):

#    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def training(epoch, model, train_pos, test_pos, batch_index, option, optimizer, alpha):
    epoch_loss = 0.0
    random.shuffle(train_pos)
    total_num = len(train_pos)
    for iteration in range(len(batch_index)):
        index1 = iteration * option.batch_size
        index2 = batch_index[iteration]
        # batch_x,batch_y,batch_loc= Reader().get_batch_gaussian_map('train',index1, index2,train_pos,test_pos)
        batch_x, batch_y, batch_loc = Input_fun.get_batch_disp_gaussian_map_vs_curve('train', index1, index2, train_pos,test_pos)
        if torch.cuda.is_available():
            inputs = torch.Tensor(batch_x).cuda()
            targets = torch.Tensor(batch_y).cuda()
        else:
            inputs = torch.Tensor(batch_x)
            targets = torch.Tensor(batch_y)
        optimizer.zero_grad()
        outputs = model(inputs)
        # cost function
        # loss = loss_fn(alpha,model,outputs, targets)
        loss = myloss(outputs, targets)
        loss.backward()
        optimizer.step()
        # print statistics
        epoch_loss += loss.item()
        if iteration % 20 == 19:
            num = index2
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, num, total_num,
                                                                           100. * num / total_num, loss.item()))
    norm_weight = extract_weights(model)
    average_loss = epoch_loss / len(batch_index)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f} ".format(epoch, average_loss))
    return loss.item(), norm_weight


def test(epoch, model_cpu, test_pos, train_pos, alpha, option):
    model_cpu.eval()
    random.shuffle(test_pos)
    test_x, test_y, vel_loc = Input_fun.get_batch_disp_gaussian_map_vs_curve('test', 0, 0, train_pos, test_pos)
    loss_hist = []
    test_loss = 0
    vs_label_pred = []
    res_total = []
    rms_count = 0
    if option.plot:
        fig = plt.figure(num=1, figsize=(12, 8), dpi=80, clear=True)

    for i in range(len(test_x)):
        # velocity axis
        label = torch.Tensor(test_y[i])
        label = label.view([1, label.size(0)])
        # dispersion axis
        input = torch.Tensor(test_x[i])
        input = input.view([1, input.size(0), input.size(1), input.size(2)])
        # compute output

        output = model_cpu(input)  # output[batchsize,H,W]
        test_loss += myloss(output, label).item()  # sum up batch loss
        # collect output loss, need to delete
        loss_hist.append(myloss(output, label).item())
        output = output.view([output.size(1)]).data.numpy()
        vel_pred = output
        vel_syn = test_y[i]
        vs_label_pred.append([vel_pred, vel_syn])
        # np.concatenate((vel_pred,vel_syn),axis=0)
        # vs_label_pred = np.hstack((vel_pred[:, 1], vel_syn[:, 1]))
        res = (vel_pred - vel_syn)
        res_total.append(res)
        rms_count = rms_count + np.sqrt(np.power(res, 2).sum() / len(res))
        if option.plot:
            if i % 400 is 0:
                #             if True:
                # print(vel_loc[i].split('_'))
                # lat, lon = vel_loc[i].split('_')
                # lat = float(lat)
                # lon = float(lon)
                plt.plot(test_y[i], np.arange(0, len(output), 1)*0.5, '-.', color='red')
                plt.plot(output, np.arange(0, len(output), 1)*0.5, '-', color='green')
                plt.title('True')
                plt.xlabel('Vs(km/s)')
                plt.ylabel('Depth(km)')
                plt.gca().invert_yaxis()
                plt.savefig('./Figs/Fitting_epoch' + str(epoch) + '_' + vel_loc[i] + '.png', dpi=300)
                plt.pause(0.01)
                fig.clear()
                pass

    average_loss = test_loss / len(test_x)
    rms_count = rms_count / len(test_x)
    loss_hist = np.array(loss_hist)
    res_total = np.array(res_total)
    vs_label_pred = np.array(vs_label_pred)
    print("===> Avg. test loss: {:.4f} {:.4f}  ".format(average_loss, rms_count))
    return average_loss, rms_count, loss_hist, res_total, vs_label_pred


def checkpoint(epoch, model):
    torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % ('./model_para', epoch))
    print("Checkpoint saved to {}".format('%s/model_epoch_%d.pth' % ('./model_para', epoch)))


def main():
    os.system("test -d output|| mkdir output")
    os.system("test -d Figs|| mkdir Figs")
    os.system("test -d model_para|| mkdir model_para")
    writer = SummaryWriter()
    option = Config()
    alpha = option.alpha
    # fixed seed, because pytoch initilize weight randomly

    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(option.seed)
        print("cuda is available")
    else:
        print("cuda is not available")
        torch.manual_seed(option.seed)
    print('===> Loading datasets')

    batch_index, train_pos, test_pos = Reader().get_batch_file()
    # Saving test information:
    # with open('testinfo.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([batch_index, train_pos, test_pos], f)

    print('===> Building net')
    model = Net(image_width=17,
                image_height=60,
                image_outwidth=301,
                image_outheight=1,
                inchannel=2, outchannel=4)
    # model = Unet(in_ch=2, out_ch=1,image_len=17,image_len_out=13)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model,(2,17,60))
    print(model)
    # write out network structure
#    dummy_input = torch.rand(13, 2, 17, 60)
    # ouput network structure
#    with SummaryWriter(comment='Net') as w:
#        w.add_graph(model, (dummy_input))
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
    print('===> Training Net')
    time_loc = list(map(str, list(time.localtime())))
    time_loc = '_'.join(time_loc[0:-5])
    f = open('output/epochInfo' + time_loc + '.txt', 'w')
    for epoch in range(option.start, option.start + option.nEpochs + 1):
        # adjust learning rate
        # optimizer = adjust_learning_rate(option.lr,optimizer, epoch)
        # training
        tloss, norm_weight = training(epoch, model, train_pos, test_pos, batch_index, option, optimizer, alpha)
        # validation
        model.cpu()
        vloss, vrms, vloss_hist, res_total, vs_label_pred = test(epoch, model, test_pos, train_pos, alpha, option)
        model.to(device)
        
        # write log file
        writer = writerlogfile(writer, norm_weight, epoch, tloss, vloss, vrms, vloss_hist)
        if epoch % 5 is 0:
            checkpoint(epoch, model)
        elif epoch == 1:
            checkpoint(epoch, model)
        with open('./output/HistVal_TestVel_LabelVel_' + time_loc + 'epoch_' + str(epoch) + '.pkl', 'wb') as ff:
            pickle.dump([res_total, vs_label_pred], ff)
        string_output = "{}".format("%d %10.7f %10.7f %10.7f %s" % (epoch, tloss, vloss, vrms, '\n'))
        f.write(string_output)
    f.close()
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    print('Finished Training')
    checkpoint(epoch, model)


if __name__ == '__main__':
    main()
    pass
