""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64 * factor, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# sigmoid output unet
class sUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(sUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64 * factor, bilinear)
        # sigmoid output
        self.outc = sOutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# 2 stack unet
class stackUnet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(stackUnet2, self).__init__()  
        
        self.unet1=UNet(n_channels,1)        
        self.up= Up(2, 2, bilinear='True')   
        self.unet2=UNet(1,n_classes)
        

    def forward(self, x):
        z=self.unet1(x)
        z=self.up(z,x)
        z=self.unet2(z)
        return z
    
# 3 stack unet
class stackUnet3(nn.Module):
    def __init__(self,n_channels, n_classes, bilinear=True):
        super(stackUnet3, self).__init__()  
        
        self.unet1=UNet(n_channels,1)        
        self.up1= Up(2, 2, bilinear='True') 
        self.unet2=UNet(1,1)        
        self.up2= Up(2, 2, bilinear='True') 
        self.unet3=UNet(1,n_classes)
        

    def forward(self, x):
        z=self.unet1(x)
        z=self.up1(z,x)
        z=self.unet2(z)
        z=self.up2(z,x)
        z=self.unet3(z)
        return z
    
# 4 stack unet

class stackUnet4(nn.Module):
    def __init__(self,n_channels, n_classes, bilinear=True):
        super(stackUnet4, self).__init__()  
        
        self.unet1=UNet(n_channels,1)        
        self.up1= Up(2, 2, bilinear='True') 
        self.unet2=UNet(1,1)        
        self.up2= Up(2, 2, bilinear='True') 
        self.unet3=UNet(1,1)
        self.up3= Up(2, 2, bilinear='True') 
        self.unet4=UNet(1,n_classes)
        

    def forward(self, x):
        z=self.unet1(x)
        z=self.up1(z,x)
        z=self.unet2(z)
        z=self.up2(z,x)
        z=self.unet3(z)
        z=self.up3(z,x)
        z=self.unet4(z)
        return z
    