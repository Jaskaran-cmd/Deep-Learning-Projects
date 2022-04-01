
from turtle import forward
import torch
print(torch.cuda.is_available())
import torch.nn as nn

from discriminator import Discriminator 

class ConvBlock(nn.Module):

    def __init__(self,in_channels,out_channels,activation='relu',down=True,use_dropout=False):

        super().__init__()
        self.use_dropout=use_dropout
        self.seq=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1)
            if down==True else nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if activation=='relu' else nn.LeakyReLU(0.2)


        )

        self.dropout=nn.Dropout(0.2)

    def forward(self,x):
        out=self.seq(x)

        return out if self.use_dropout==False else self.dropout(out)

class Generator(nn.Module):

    def __init__(self,in_channel,out_channel):
        super().__init__()

        self.init_layer=nn.Sequential(
            nn.Conv2d(in_channels=in_channel*2,out_channels=out_channel,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2)   ############OUTPUT_SHAPE=(BATCH_SIZE,64,128,128)
        )

        self.d1=ConvBlock(out_channel,out_channel*2,activation='leak',use_dropout=False) #(BATCH_SIZE,128,64,64)
        self.d2=ConvBlock(out_channel*2,out_channel*4,activation='leaky',use_dropout=False) #(BATCH_SIZE,256,32,32)
        self.d3=ConvBlock(out_channel*4,out_channel*2*4,activation='leaky',use_dropout=False) #(BATCH_SIZE,512,16,16)
        self.d4=ConvBlock(out_channel*2*4,out_channel*2*4,activation='leaky',use_dropout=False) #(BATCH_SIZE,512,8,8)
        self.d5=ConvBlock(out_channel*2*4,out_channel*2*4,activation='leaky',use_dropout=False) #(BATCH_SIZE,512,4,4)

        self.bottleneck=nn.Sequential(
            nn.Conv2d(out_channel*2*4,out_channel*2*4,kernel_size=4,stride=1,padding=0),
            nn.BatchNorm2d(out_channel*2*4),
            nn.LeakyReLU(0.2)
            
                )


        self.u1=nn.Sequential(
            nn.ConvTranspose2d(out_channel*2*4,out_channel*2*4,kernel_size=4,stride=1,padding=0),
            nn.BatchNorm2d(out_channel*2*4),
            nn.LeakyReLU(0.2)
            
                ) #(BATCH_SIZE,512,4,4)
        #u1 cat d5  in shape (1024,4,4)
        self.u2=ConvBlock(out_channel*2*4*2,out_channel*2*4,down=False,activation='relu',use_dropout=True) #(512,8,8)
        #u2 cat d4   in_shape__>(1024,8,8)
        self.u3=ConvBlock(out_channel*2*4*2,out_channel*2*4,down=False,activation='relu',use_dropout=True) #(512,16,16)
        #u3 cat d3   in_shape___>(1024,16,16)
        self.u4=ConvBlock(out_channel*2*4*2,out_channel*4,down=False,activation='relu',use_dropout=True)     #(256,32,32)
        #u4 cat d2   in_shape___>(512,32,32)
        self.u5=ConvBlock(out_channel*4*2,out_channel*2,down=False,activation='relu',use_dropout=True)       #(128,64,64)
        #u5 cat d1   in shape___>(256,64,64)
        self.u6=ConvBlock(out_channel*2*2,out_channel,down=False,activation='relu',use_dropout=True)         #(64,128,128)
        #u6 cat u0   in_shape(128,128,128)
        self.final=ConvBlock(out_channel*2,3,down=False,use_dropout=True)                                    #(3,256,256)

    def forward(self,x):

        #####################________ENCODER_______##########################
        d0=self.init_layer(x)
        #SHAPE___________________________>(1,64,128,128)
        d1=self.d1(d0)
        #SHAPE___________________________>(1,128,64,64)
        d2=self.d2(d1)
        #SHAPE___________________________>(1,256,32,32)
        d3=self.d3(d2)
        #SHAPE___________________________>(1,512,16,16)
        d4=self.d4(d3)
        #SHAPE___________________________>(1,512,8,8)
        d5=self.d5(d4)
        #SHAPE___________________________>(1,512,4,4)
        ###################________BOTTLENECK________###############################
        bot=self.bottleneck(d5)
        #SHAPE___________________________>(1,512,1,1)



        ###################________DECODER__________###############################

        u1=self.u1(bot)
        #SHAPE__________________________>(1,512,4,4)
        u2=self.u2(torch.cat([u1,d5],dim=1))
        #SHAPE__________________________>(1,512,8,8)
        u3=self.u3(torch.cat([u2,d4],dim=1))
        #SHAPE__________________________>(1,512,16,16)
        u4=self.u4(torch.cat([u3,d3],dim=1))
        #SHAPE__________________________>(1,256,32,32)
        u5=self.u5(torch.cat([u4,d2],dim=1))
        #SHAPE__________________________>(1,128,64,64)
        u6=self.u6(torch.cat([u5,d1],dim=1))
        #SHAPE__________________________>(1,64,128,128)
        u7=self.final(torch.cat([u6,d0],dim=1))

        return u7

        



        



if __name__=="__main__":

    convout=Generator(3,64).cuda()
    randin=torch.randn((30,3,256,256)).cuda()
    print(convout(randin).shape)


    
        
        
        