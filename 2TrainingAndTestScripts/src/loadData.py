#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : loadData.py
@Description :
@Time    : 2020/11/23 14:02:01
@Author  : Jing Hu
@Email   : jhsa920@163.com
@Version : 1.0
'''
import os
import numpy as np
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import DataLoader,Dataset
import torch   
def get_Files(dispPath,vsPath=None):
    """
	dispPath: path for dispersion curves (training or test)
	vsPath: if vsPath is None, it will read real dispersion curves.
		    else vsPath is not None, it will read training dispersion curves.
    """
    filename_dispersion_total = []
    filename_vs_total = []
    key_total = [] # geographical location
    if os.path.exists(dispPath): #
        filename_all = os.listdir(dispPath)
        if vsPath is None: # read real disp files
            for filename in filename_all:
                key = list(map(float,filename[0:-4].split('_'))) 
                disp_file = dispPath + filename
                if os.path.exists(disp_file):
                    filename_dispersion_total.append(disp_file)
                    key_total.append(key)
            filename_dispersion_total = np.array(filename_dispersion_total)
            key_total = np.array(key_total)
        elif os.path.exists(vsPath):  #read train file name
            for filename in filename_all:
                key = list(map(float,filename[2:-4].split('_')))   # add group disp and phase disp # lon lat float 
                disp_file = dispPath + filename
                vs_file   = vsPath +   filename
                if os.path.exists(disp_file) and os.path.exists(vs_file):
                    filename_dispersion_total.append(disp_file)
                    filename_vs_total.append(vs_file)
                    key_total.append(key)
            filename_dispersion_total = np.array(filename_dispersion_total)
            filename_vs_total = np.array(filename_vs_total)
            key_total = np.array(key_total)

    else:
        print('Input train file path is not exist, check the input path!')
        filename_dispersion_total=None;filename_vs_total=None;key_total=None
    # for training, those files returned are used to split train and validation dataset.
    return filename_dispersion_total, filename_vs_total, key_total
    
def train_validSplit(filenames,validSize=0.2):
    """
    filenames: a np.array include many filenames. np.array(['../a.jpg','../b.jpg','../c.jpg']
    validSize: validation portion. 
    """
    trainIndex, validIndex= train_test_split(range(len(filenames)), test_size=validSize, random_state=42)
    #random.shuffle(trainIndex); random.shuffle(validIndex)
    return trainIndex, validIndex

def getTrainValidationFiles(dispPath,vsPath,validSize=0.2):
    """
	dispPath: path for dispersion curves (training or test)
	vsPath: if vsPath is None, it will read real dispersion curves.
		    else vsPath is not None, it will read training dispersion curves.
	return: trainFiles,validFiles
	"""
    dispFiles,vsFiles, locKeys=get_Files(dispPath,vsPath)
    trainIndex, validIndex=train_validSplit(dispFiles,validSize)
    trainFiles={'disp':dispFiles[trainIndex],'vs':vsFiles[trainIndex],'location':locKeys[trainIndex]}
    validFiles={'disp':dispFiles[validIndex],'vs':vsFiles[validIndex],'location':locKeys[validIndex]}
    return trainFiles,validFiles
def readRealData(dispPath):
    # get real dispersion files
    filename_dispersion_total, _, key_total=get_Files(dispPath,vsPath=None)
    totalRealData=[]
    for filenameDisp in filename_dispersion_total:
        dataDisp = np.loadtxt(filenameDisp).astype(np.float32)
        totalRealData.append(dataDisp)
    return np.array(totalRealData),key_total
def readTrainingData(filenameDisp,filenameVs):
    """
	filename:  
	data_type: "disp" for dispersion curves; "vs" for 
    """
    dataDisp = np.load(filenameDisp).astype(np.float32)  # load xx.npy file not txt file
    dataVs   = np.load(filenameVs)[:,1].astype(np.float32)  # 
    return dataDisp, dataVs

# create disp-vs dataset
class DispVsDataset(Dataset):
    '''
    Reference https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    '''
    def __init__(self,Files,transform=None):
        """
    	Files: a dict, like trainFiles={'disp':dispFiles[trainIndex],'vs':vsFiles[trainIndex],'location':locKeys[trainIndex]}  
        """
    	# get dispersion files, vs files, location key
        self.dispFiles = Files['disp']
        self.vsFiles = Files['vs']
        self.locKeys = Files['location']
        self.transform=transform
    def __len__(self):
        return len(self.dispFiles)
    def __getitem__(self, idx):
        disp,vs=readTrainingData(self.dispFiles[idx],self.vsFiles[idx])  
        sample = {'disp': disp,'vs':vs,'location':self.locKeys[idx]}
        if self.transform:
            sample=self.transform(sample)
        return sample
class ToTensor(object):
    """
    convert ndarrays in sample to Tensors
    """
    def __call__(self,sample):
        disp,vs=torch.from_numpy(sample['disp']),torch.from_numpy(sample['vs'])
        location=torch.from_numpy(sample['location'])
        return {'disp':disp,'vs':vs,'location':location}
if  __name__ == "__main__":
    from config import Config
    import matplotlib.pyplot as plt
    dispPath=Config().filepath_disp_training
    batchsize=Config().batch_size
    vsPath=Config().filepath_vs_training
    trainFiles,validFiles=getTrainValidationFiles(dispPath,vsPath,validSize=0.2)
    trainDataset=DispVsDataset(trainFiles,transform=ToTensor())
    validDataset=DispVsDataset(validFiles,transform=ToTensor())
    sample=trainDataset[1]

    trainLoader = DataLoader(trainDataset, batch_size=batchsize,
                        shuffle=True, num_workers=0)
    validLoader=DataLoader(validDataset, batch_size=batchsize,
                        shuffle=True, num_workers=0)
    
    for i_batch, sample_batched in enumerate(trainLoader):
        print(i_batch, sample_batched['disp'].size(),
            sample_batched['vs'].size())
        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            plt.subplot(2,2,1)
            plt.imshow(sample_batched['disp'][0,0,:,:].data.numpy())
            plt.subplot(2,2,3)
            plt.imshow(sample_batched['disp'][0,1,:,:].data.numpy())
            plt.subplot(2,2,(2,4))
            plt.plot(sample_batched['vs'][0,:].data.numpy())
            plt.show()
            break

    print(validDataset)
