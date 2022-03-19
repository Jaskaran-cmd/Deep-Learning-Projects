
import torch 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import os
class AnimeData(Dataset):
    def __init__(self,root,transforms=None):
        self.transforms=transforms
        self.root=root
        self.images=os.listdir(self.root)

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self,idx):
        image=self.images[idx]
        image=Image.open(os.path.join(self.root,image))
        if self.transforms:

            image=self.transforms(image)

        return image 

def data_loader():
    image_size=64
    batch_size=100
    num_workers=12
    shuffle=True
    pin_memory=True
    normalize=((0.5,0.5,0.5),(0.5,0.5,0.5))
    root_dir='D:\VScode\images'
    transform=transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
        transforms.CenterCrop(image_size)
    ]
    )
    dataset=AnimeData(root=root_dir,transforms=transform)
    dataloader=DataLoader(dataset,shuffle=shuffle,pin_memory=pin_memory,batch_size=batch_size)

    return dataset,dataloader

def show_images(image,max_size=64):
    stats=((0.5,0.5,0.5),(0.5,0.5,0.5))
    fig,ax=plt.subplots(figsize=(8,8))
    ax.set_xticks([]);ax.set_yticks([])
    def denorm(image,stats):
        return image*stats[1][0] + stats[0][0]
    ax.imshow(make_grid(denorm(image.detach(),stats),nrow=8).permute(1,2,0))
    plt.show()

def show_batch(dl,max_size=64):
    for image in dl:
        show_images(image,max_size)
        break

def denorm(image,stats):
    return image*stats[1][0] + stats[0][0]

    

if __name__=='__main__':
    dataset,datalaoder=data_loader()
    # for images in datalaoder:
    #     max_size=64
    #     stats=((0.5,0.5,0.5),(0.5,0.5,0.5))
    #     fig,ax=plt.subplots(figsize=(8,8))
    #     ax.set_xticks([]); ax.set_yticks([])
    #     # images=images*stats[1][0] + stats[0][0]
    #     ax.imshow(make_grid(images.detach(),nrow=8).permute(1,2,0))
    #     print(images.shape)
    #     plt.show()2
    
    #     break
    xb = torch.randn(100, 128, 1, 1) 
    model=torch.nn.ConvTranspose2d(128,512,kernel_size=(4,4),stride=1,padding=0)
    out=model(xb)
    print(torch.ones(100,1,1))
    print(next(iter(datalaoder)).shape[0])
    


    
    
