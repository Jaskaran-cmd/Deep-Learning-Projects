import torch 
from torch.utils.data import DataLoader,Dataset
import os 
import numpy as np 
import cv2 as cv 
import albumentations as A
import matplotlib.pyplot as plt 


from albumentations.pytorch import ToTensorV2
class Pix2PixData(Dataset):

    def __init__(self,root_dir,transforms):
        self.root_dir=root_dir
        self.all_files=os.listdir(self.root_dir)
        self.transforms=transforms
        

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        curr_path=self.all_files[idx]
        image=cv.imread(os.path.join(self.root_dir,curr_path))
        in_image,target_image=image[:,512:,],image[:,:512,:]
        if self.transforms:
            transformed=self.transforms(image=in_image,mask=target_image)
            in_image=transformed['image']
            target_image=transformed['mask']

        return in_image,target_image


def loader(root_dir,transforms,batch_size,num_workers):
    dataset=Pix2PixData(root_dir=root_dir,transforms=transforms)
    dataloader=DataLoader(dataset=dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True,pin_memory=True)
    return dataset,dataloader



    
if __name__=='__main__':
    root_dir='D:/VScode/Pix2Pix/AnimeData/data/train'
    transforms=A.Compose([
        A.Resize(256,256),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20),
        ToTensorV2()
        
    ])
    data=Pix2PixData(root_dir=root_dir,transforms=transforms)

    image,mask=data[20]
    print(image.shape)
    print(mask.shape)
    plt.imshow(mask)   
    # plt.imshow(image.permute(1,2,0))
    print(len(data))
    plt.show()







