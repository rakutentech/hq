from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import random
import json



from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature

from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils import data
from PIL import Image



#####################################
####### network architechture########
####################################
import torchvision.models as models

class Net1(nn.Module):
    def __init__(self,feature_dim):
        super(Net1, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        num_ftrs = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(4096,feature_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(1000, 4096), #nus-wide
            nn.Dropout(), 
            nn.ReLU(),
            nn.Linear(4096,feature_dim),
            
        )

    def forward(self, image,text):
        #image size 227, out_dim = 128        
        img = self.alexnet(image)
        txt = self.mlp(text)
        
        return img,  txt 

class Net2(nn.Module):
    def __init__(self,feature_dim):
        super(Net2, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        num_ftrs = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(4096,feature_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(1386, 4096),
            nn.Dropout(), 
            nn.ReLU(),
            nn.Linear(4096,feature_dim),
            
        )

    def forward(self, image,text):
        #image size 227, out_dim = 128
        
        img = self.alexnet(image)
        txt = self.mlp(text)
        return img,  txt 
