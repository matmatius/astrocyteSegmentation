#Use the numpy library.

import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


from skimage import data
from skimage import filters
from skimage import exposure



# These are the labels we predicted.
#pred_labels = otsuOut
#print ('pred labels:\t\t', pred_labels)
 
# These are the true labels.
#true_labels = valset[index]['mask'][0].numpy()
#print ('true labels:\t\t', true_labels)


# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.

def truePos(pred_labels, true_labels):
    
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))  
    
    return TP
 

# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.

def trueNeg(pred_labels, true_labels):
    
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    
    return TN
    

# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.

def falsePos(pred_labels, true_labels):
    
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    
    return FP

# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.

def falseNeg(pred_labels, true_labels):
    
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    
    return FN


# Sensitivity

def sensitivity(pred_labels, true_labels):
    
    TP=truePos(pred_labels, true_labels)
    FN=falseNeg(pred_labels, true_labels)
    S=TP/(TP+FN)
    
    return S

# Precision

def precision(pred_labels, true_labels):
    TP=truePos(pred_labels, true_labels)
    FP=falsePos(pred_labels, true_labels)
    P=TP/(TP+FP)
    return P

# Dice Coefficient

def diceCoef(pred_labels, true_labels):
    TP=truePos(pred_labels, true_labels)
    FP=falsePos(pred_labels, true_labels)
    FN=falseNeg(pred_labels, true_labels)
    DC=2*TP/(2*TP+FP+FN)
        
    return DC

        

    
