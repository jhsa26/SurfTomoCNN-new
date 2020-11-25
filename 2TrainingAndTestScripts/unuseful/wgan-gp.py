#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@Time   22/05/2018 3:30 PM 2018
@Author HJ@USTC
@Email  jhsa26@mail.ustc.edu.cn
@Blog   jhsa26.github.io
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.autograd as autograd
import torch.optim as optim
import utils
import tqdm
EPSILON = 0.0001
class Discriminator(nn.Module):
    def __init__(self,image_size, channel_sizeIn, channel_sizeOut):
        # image_size means the length of signal
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.channel_sizeIn = channel_sizeIn
        self.channel_size = channel_sizeOut
        # layers
        self.conv1 = nn.Conv2d(
            channel_sizeIn,channel_sizeOut,
            kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            channel_sizeOut, channel_sizeOut*2,
            kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            channel_sizeOut*2, channel_sizeOut*4,
            kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            channel_sizeOut*4, channel_sizeOut*8,
            kernel_size=3, stride=1, padding=1,
        )
        self.fc = nn.Linear(image_size * channel_sizeOut*8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1,self.image_size* self.channel_sizeOut*8) #?
        return self.fc(x)

class Generator(nn.Module):
    def __init__(self,z_size, image_size, channel_sizeIn, channel_sizeOut):
        super(Generator, self).__init__()
        # configurations
        self.z_size = z_size
        self.image_size = image_size
        self.channel_sizeIn = channel_sizeIn
        self.channel_sizeOut = channel_sizeOut

        # layers
        self.fc = nn.Linear(z_size,image_size*channel_sizeOut*8)

        self.bn0 = nn.BatchNorm2d(channel_sizeOut*8)

        self.bn1 = nn.BatchNorm2d(channel_sizeOut*4)
        self.deconv1 = nn.ConvTranspose2d(
            channel_sizeOut*8, channel_sizeOut*4,
            kernel_size=3, stride=1, padding=1
        )

        self.bn2 = nn.BatchNorm2d(channel_sizeOut*2)
        self.deconv2 = nn.ConvTranspose2d(
            channel_sizeOut*4, channel_sizeOut*2,
            kernel_size=3, stride=1, padding=1,
        )

        self.bn3 = nn.BatchNorm2d(channel_sizeOut)
        self.deconv3 = nn.ConvTranspose2d(
            channel_sizeOut*2, channel_sizeOut,
            kernel_size=3, stride=2, padding=1
        )

        self.deconv4 = nn.ConvTranspose2d(
            channel_sizeOut, channel_sizeIn,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        g = self.fc(z)
        g=g.view(z.size(0), self.channel_sizeOut*8 , 1 , self.image_size)

        g = F.relu(self.bn0(g))
        g = F.relu(self.bn1(self.deconv1(g)))
        g = F.relu(self.bn2(self.deconv2(g)))
        g = F.relu(self.bn3(self.deconv3(g)))
        g = self.deconv4(g)
        return F.sigmoid(g)

#WGAN主要从损失函数的角度对GAN做了改进，损失函数改进之后的WGAN即使在全链接层上也能得到很好的表现结果，WGAN对GAN的改进主要有：

#判别器最后一层去掉sigmoid

#生成器和判别器的loss不取log

#对更新后的权重强制截断到一定范围内，比如[-0.01，0.01]，以满足论文中提到的lipschitz连续性条件。

#论文中也推荐使用SGD， RMSprop等优化器，不要基于使用动量的优化算法，比如adam，但是就我目前来说，训练GAN时，我还是adam用的多一些。
#充分利用神经网络拟合性能以及数值近似。

#判别器最后一层去掉sigmoid（使得loss更明显）；
#生成器和判别器的loss不去log（使得loss更明显）；
#判别器的参数更新截断；
#不用基于动量的优化算法。

class WGANGP(nn.Module):
    def __init__(self, label, z_size,
                 image_size, channal_SizeIn,
                 d_channel_size, g_channel_size):
        # configurations
        super().__init__()
        self.label = label
        self.z_size = z_size
        self.image_size = image_size
        self.channal_SizeIn = channal_SizeIn
        self.d_channel_size = d_channel_size
        self.g_channel_size = g_channel_size

        # components
        self.discriminator = Discriminator(image_size=self.image_size,channal_SizeIn=self.channal_SizeIn ,
                                           channal_SizeOut=self.d_channel_size,)
        self.generator = Generator(z_size=self.z_size,image_size=self.image_size,
                                   image_channel_size=self.image_channel_size,channal_SizeOut=self.g_channel_size)
    @property
    def name(self):
        return (
            'WGAN-GP'
            '-z{z_size}'
            '-c{c_channel_size}'
            '-g{g_channel_size}'
            '-{label}-{image_size}x{image_size}x{image_channel_size}'
        ).format(
            z_size=self.z_size,
            c_channel_size=self.d_channel_size,
            g_channel_size=self.g_channel_size,
            label=self.label,
            image_size=self.image_size,
            image_channel_size=self.channal_SizeIn,
        )

    def c_loss(self, x, z, return_g=False):
        g = self.generator(z)
        d_x = self.discriminator(x).mean()
        d_g = self.discriminator(g).mean()
        l = -(d_x - d_g)
        return (l, g) if return_g else l

    def g_loss(self, z, return_g=False):
        g = self.generator(z)
        l = -self.discriminator(g).mean()
        return (l, g) if return_g else l

    def sample_image(self, size):
        return self.generator(self.sample_noise(size))

    def sample_noise(self, size):
        z = torch.randn(size, self.z_size)* .1
        return z                            #.cuda() if self._is_on_cuda() else z

    def gradient_penalty(self, x, g, lamda):
        assert x.size() == g.size()
        a = torch.rand(x.size(0), 1)
        a = a                                   #.cuda() if self._is_on_cuda() else a
        #x.nelement  means total elements in x tensor
        a = a.expand(x.size(0), x.nelement() // x.size(0))\
            .contiguous()\
            .view(x.size(0),
                  self.channal_SizeIn,
                  self.image_size,
                  self.image_size)

        interpolated = a * x.data + (1 - a) * g.data
        c = self.discriminator(interpolated)
        gradients = autograd.grad(
            c, interpolated, grad_outputs=(torch.ones(c.size()),torch.ones(c.size())),create_graph=True,retain_graph=True,)[0]
        return lamda * ((1 - (gradients + EPSILON).norm(2, dim=1)) ** 2).mean()

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda






if __name__ == '__main__':
    pass