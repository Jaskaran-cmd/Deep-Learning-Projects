import torch
import torch.nn as nn 
import albumentations as A 
from torch.cuda.amp import GradScaler 
from discriminator import Discriminator
from generator import Generator
from dataset import loader
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import os 
from torchvision.utils import make_grid,save_image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.nn import L1Loss

l1_loss=L1Loss()

def denorm(image,stats):
    return image*stats[0][0]+stats[0][1]

def save_sample(index,latent_tensor,show=True):
    root='D:\VScode\WganGenerated'
    fake_images=generator(latent_tensor)
    stats=((0.5,0.5,0.5),(0.5,0.5,0.5))
    fake_fname=f'generated_image{index}.png'
    save_image(denorm(fake_images,stats),os.path.join(root,fake_fname),nrow=8)
    
    if show==True:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
        plt.show()



if __name__=='__main__':
    lr=3e-4
    NUM_EPOCHS=20
    BATCH_SIZE=10
    NUM_WORKERS=12
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir='D:/VScode/Pix2Pix/AnimeData/data/train'

    transforms=transforms=A.Compose([
        A.Resize(256,256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2()
        
    ])
    dataset,dataloader=loader(root_dir=root_dir,transforms=transforms,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
    scaler1=GradScaler()
    scaler2=GradScaler()
    generator=Generator(3,64).to(DEVICE)
    discriminator=Discriminator(3).to(DEVICE)
    gen_optim=torch.optim.Adam(generator.parameters(),lr)
    des_optim=torch.optim.Adam(discriminator.parameters(),lr)
    loss_fn=nn.CrossEntropyLoss()
    writer=SummaryWriter()
    step=0
    

    for epoch in range(NUM_EPOCHS):

        for batch,(image,mask) in enumerate(dataloader):

            image,mask=image.to(DEVICE),mask.to(DEVICE)

            with torch.cuda.amp.autocast():

                f_image=generator(image)
                d_pred=discriminator(image,mask)
                loss_1=loss_fn(d_pred,torch.ones_like(d_pred))
                d_pred_fake=discriminator(image,f_image.detach())
                loss_2=loss_fn(d_pred_fake,torch.zeros_like(d_pred_fake))
                t_loss=loss_1+loss_2

            des_optim.zero_grad()
            scaler1.scale(t_loss).backward()
            scaler1.step(des_optim)
            scaler1.update()


            with torch.cuda.amp.autocast():

                fake_out=discriminator(image,f_image)
                d_pred_fake=loss_fn(fake_out,torch.ones_like(fake_out))
                l1_loss=l1_loss(fake_out,mask)*100
                t1_loss=d_pred_fake+l1_loss

            gen_optim.zero_grad()
            scaler2.scale(t1_loss).backward()
            scaler2.step(gen_optim)
            scaler2.update()
            if batch==30:
                writer.add_scalar('generator_loss',t1_loss,global_step=step)
                writer.add_scalar('discriminator_loss',t_loss,global_step=step)
                image_grid1=make_grid(image)
                image_grid2=make_grid(fake_out)
                image_grid3=make_grid(mask)
                writer.add_image('Anime_images',image_grid1,global_step=step)
                writer.add_images('fake_mask',image_grid2,global_step=step)
                writer.add_images('mask',image_grid3,global_step=step)

            step+=1

            save_sample(batch+1,image,show=False)
    
    path='myModel.pt'
    torch.save({
        'des_state':discriminator.state_dict(),
        'generator_state':generator.state_dict(),
        'des_optim_state':des_optim.state_dict(),
        'gen_optim': gen_optim.state_dict(),
    },path)


            





            




    

    


    

