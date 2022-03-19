import  torch
import os 
import torch.nn as nn 
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from Data import data_loader
from GANs import Generator,Discriminator
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torchvision.utils import save_image
from Data import denorm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt 
#HYPER PARAMETERs 
Scaler=GradScaler()
randn_channel=128

device= 'cuda' if torch.cuda.is_available() else 'cpu'
fixed_latent = torch.randn(100, 128, 1, 1, device=device)
Descriminator=Discriminator().to(device)
generator=Generator(randn_channel).to(device)
def train_descriminator(real_image,optim_d):
    optim_d.zero_grad()
    desc_out=Descriminator(real_image)
    real_target=torch.ones(real_image.size(0),1,device=device)
    real_loss=F.binary_cross_entropy(desc_out,real_target)


    noise=torch.randn(100, 128, 1, 1).to(device)
    fake_image=generator(noise)
    fake_out=Descriminator(fake_image)
    fake_target=torch.zeros(fake_image.size(0),1,device=device)
    fake_loss=F.binary_cross_entropy(fake_out,fake_target)

    total_loss=real_loss+fake_loss
    total_loss.backward()
    optim_d.step()

    return total_loss.item()

def train_generator(optim_g):
    
    optim_g.zero_grad()
    noise=torch.randn(100, 128, 1, 1).to(device)
    fake_image=generator(noise)
    fake_out=Descriminator(fake_image)
    fake_target=torch.ones(fake_image.size(0),1,device=device)

    fake_loss=F.binary_cross_entropy(fake_out,fake_target)

    fake_loss.backward()
    optim_g.step()
    return fake_loss

def save_sample(index,latent_tensor,show=True):
    root='D:\VScode\GeneratedImages'
    fake_images=generator(latent_tensor)
    stats=((0.5,0.5,0.5),(0.5,0.5,0.5))
    fake_fname=f'generated_image{index}.png'
    save_image(denorm(fake_images,stats),os.path.join(root,fake_fname),nrow=8)
    
    if show==True:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
        plt.show()






def train():

    lr=0.0001
    num_epochs=20
    _,dataloader=data_loader()
    desc_opt=torch.optim.Adam(Descriminator.parameters(),lr=lr)
    gen_opt=torch.optim.Adam(generator.parameters(),lr=lr)
    fixed_latent = torch.randn(100, 128, 1, 1, device=device)
    loss_desc=[]
    loss_gen=[]
    for epoch in range(num_epochs):
        for batch,real in enumerate(tqdm(dataloader)):
            real=real.to(device)
            
            desc_loss=train_descriminator(real,desc_opt)
            gen_loss=train_generator(gen_opt)


        loss_desc.append(desc_loss)
        loss_gen.append(gen_loss)
        print(f'epoch:[{epoch}] descriminator_loss:[{desc_loss}] generator_loss:[{gen_loss}]')

        save_sample(1+epoch,latent_tensor=fixed_latent,show=False)

    torch.save(generator.state_dict(),'G.pt')
    torch.save(Descriminator.state_dict(),'D.pt')

if __name__=='__main__':
    train()

    








