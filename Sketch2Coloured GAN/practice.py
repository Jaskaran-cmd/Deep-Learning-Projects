
import cv2 as cv
import torch 

image=cv.imread('D:/VScode/Pix2Pix/AnimeData/data/train/2971128.png')
print(torch.tensor(image).permute(2,1,0).shape)