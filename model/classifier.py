import torch.nn as nn
import torch
from model.Image_classify import *

class Classifier(ImageclassificationBase,nn.Module):
    def __init__(self,inchannel=3):
        super(Classifier,self).__init__()
        self.conv1 = nn.Conv2d(inchannel,32,kernel_size = 3, padding = 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(82944, 1024)
        self.relu7 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 512)
        self.relu8 = nn.ReLU()
        self.linear3 = nn.Linear(512, 2)

    def forward(self,x):
        #x = self.conv1(x)
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu7(x)
        x = self.linear2(x)
        x = self.relu8(x)
        x = self.linear3(x)

        return x
