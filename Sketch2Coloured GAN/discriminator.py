import torch 

import torch.nn as nn 

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride=2):
        super().__init__()
        self.seq=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=4,stride=stride,padding=0,bias=False,padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2)
        )

    def forward(self,xb):
        return self.seq(xb)


class Discriminator(nn.Module):

    def __init__(self,num_channels,features=[64,128,256,512]):
        super(Discriminator,self).__init__()
        self.features=features
        self.initial_layer=nn.Sequential(
            nn.Conv2d(in_channels=num_channels*2,out_channels=self.features[0],kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2)

        )

        layers=[]
        in_feature=self.features[0]
        for feature in self.features[1:]:

            layers.append(ConvBlock(in_channel=in_feature,out_channel=feature,stride=1 if feature==self.features[-1] else 2 ))
            in_feature=feature

        layers.append(nn.Conv2d(in_channels=in_feature,out_channels=1,stride=1,padding=1,kernel_size=4,padding_mode='reflect'))

        self.ConvBlock=nn.Sequential(*layers)

        #(3,256,256)--->(64,128,128)
        #(64,128,128)--->(128,64,64)
        #(128,64,64)---->(256,32,32)
        #(256,32,32)---->(512,)


    def forward(self,x1,x2):
        x=torch.cat([x1,x2],dim=1)
        

        out=self.initial_layer(x)
        return self.ConvBlock(out)

if __name__=='__main__':

    model=Discriminator(num_channels=3)
    sample_iput=torch.randn((100,3,256,256))
    sample_iput2=torch.randn((100,3,256,256))
    out=model(sample_iput,sample_iput2)
    print(out.shape)