from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os

from skimage.filters import gaussian # for otsu if required.

from tqdm import tqdm

from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise

class Dataset(Dataset):

    # Constructor
    #def __init__(self,transform=None,train=True):
    def __init__(self,dir_image,dir_mask):
        self.image_files=[os.path.join(dir_image,file) for file in  os.listdir(dir_image) if file.endswith(".tif")]
        self.image_files.sort()
        self.mask_files=[os.path.join(dir_mask,file) for file in  os.listdir(dir_mask)if file.endswith(".tif")] 
        self.mask_files.sort()
        self.len=len(self.image_files)
        
    # Get the length
    def __len__(self):
        return self.len
    
    def preprocess(cls,pil_img):
        
        # HWC to CHW
        # transpose for pytorch
        img_nd = np.array(pil_img)

        img_trans = img_nd.transpose((2, 0, 1))
        
                
        if img_trans.max() > 1:
            img_trans = img_trans / 255
                
        if (3,128,128)==img_trans.shape:
            img_trans=(img_trans[0]+img_trans[1]+img_trans[2])/3
            img_trans=img_trans.reshape((1,128,128))
            
        return img_trans
    # Getter
    def __getitem__(self, idx):
               
        image=self.preprocess(Image.open(self.image_files[idx]))
        mask=self.preprocess(Image.open(self.mask_files[idx]))
        
                  
        # If there is any transform method, apply it onto the image
        #if self.transform:
            #image = self.transform(image)

        return {'image': torch.from_numpy(image), 'mask':torch.from_numpy(mask).type(torch.LongTensor)}
    
# rotation methos for ground truth or mask.
def rotationMask45(x):
    rotated = rotate(x, angle=45)
    #val = filters.threshold_otsu(rotated) #this otsu might not be needed
    val=1e-19 #empirical value.
    rotated=(rotated>val)*1 #otsu
    return rotated

def rotationMask135(x):
    rotated = rotate(x, angle=135)
    #val = filters.threshold_otsu(rotated) #this otsu might not be needed
    val=1e-19 #empirical value.
    rotated=(rotated>val )*1 #otsu
    return rotated
    
def augmentData(dataset):
    listImage= []
    listMask = []
    for i in tqdm(range(len(dataset))):
        listImage.append( dataset[i]['image'])
 
        listImage.append(torch.from_numpy(rotate(dataset[i]['image'][0], angle=45)).view(1,128,128))
        listImage.append(torch.from_numpy(rotate(dataset[i]['image'][0], angle=135)).view(1,128,128))
    
        listImage.append(torch.from_numpy(np.array(np.fliplr(dataset[i]['image'][0]))).view(1,128,128))
        listImage.append(torch.from_numpy(np.array(np.flipud(dataset[i]['image'][0]))).view(1,128,128))
    
        # Not sure either include or not random noise
        listImage.append(torch.from_numpy(random_noise(dataset[i]['image'][0],var=0.2**2)).view(1,128,128))
    

                     
        #Mask
        listMask.append(dataset[i]['mask'])
        listMask.append(torch.from_numpy(rotationMask45(dataset[i]['mask'][0])).view(1,128,128))
        listMask.append(torch.from_numpy(rotationMask135(dataset[i]['mask'][0])).view(1,128,128))
        listMask.append(torch.from_numpy(np.array(np.fliplr(dataset[i]['mask'][0]))).view(1,128,128))
        listMask.append(torch.from_numpy(np.array(np.flipud(dataset[i]['mask'][0]))).view(1,128,128))
    
    
        # Not sure either include or not random noise
        listMask.append(dataset[i]['mask'])
    
    return {'image': listImage, 'mask': listMask }

class augDateSet(Dataset):
    def __init__(self, listImage, listMask):
        self.listImage = listImage
        self.listMask = listMask
        
    def __len__(self):
        return len(self.listImage)
            
    def __getitem__(self, i):

        return {'image': self.listImage[i], 'mask': self.listMask[i]}

    
    