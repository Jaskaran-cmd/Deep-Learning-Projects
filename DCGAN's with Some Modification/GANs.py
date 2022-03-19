from audioop import bias
from cmath import tanh
from re import S
from turtle import forward
import torch 
import torchvision
import torch.nn as nn 

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False)   #INPUT SIZE>OUTPUT SIZE 3X64X64--->64X32X32
        self.batch_norm=nn.BatchNorm2d(64)
        self.l_relu=nn.LeakyReLU(0.2,inplace=True)

        self.conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False) #64X32X32--->128X16X16
        self.batch_norm2=nn.BatchNorm2d(128)
        
        self.conv3=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False) #128X16X16---->256X8X8
        self.batch_norm3=nn.BatchNorm2d(256)

        self.conv4=nn.Conv2d(in_channels=256,out_channels=512,stride=2,padding=1,kernel_size=4,bias=False) #256X16X16---->512X4X4
        self.batch_norm4=nn.BatchNorm2d(512)

        self.conv5=nn.Conv2d(in_channels=512,out_channels=1,stride=1,kernel_size=4,padding=0,bias=False) #512X4X4----->1X1X1
        self.flatten=nn.Flatten()
        self.sigmoid=nn.Sigmoid()

    def forward(self,image):
        conv1=self.l_relu(self.batch_norm(self.conv1(image)))
        conv2=self.l_relu(self.batch_norm2(self.conv2(conv1)))
        conv3=self.l_relu(self.batch_norm3(self.conv3(conv2)))
        conv4=self.l_relu(self.batch_norm4(self.conv4(conv3)))
        output=self.sigmoid(self.flatten(self.conv5(conv4)))
        return output

class Generator(nn.Module):

    def __init__(self,randn_channel):
        super(Generator,self).__init__()
        self.seq=nn.Sequential(

            nn.ConvTranspose2d(in_channels=randn_channel,out_channels=512,kernel_size=4,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),########################################(512,4,4)

            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),##########################################(256,8,8)

            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1,bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),##########################################(128,16,16)

            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),###########################################(64,32,32)

            nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=4,padding=1,stride=2,bias=True),
            nn.Tanh(),############################################(3,64,64)

        )

    def forward(self,xb):
        return self.seq(xb)


        
        

