import torchvision
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import cv2

train_dir = r'C:\Users\utkar\OneDrive\Desktop\New folder\EfficientNet_Images\train'
test_dir = r'C:\Users\utkar\OneDrive\Desktop\New folder\EfficientNet_Images\test'

transform = transforms.Compose([transforms.Resize((150,150)),transforms.ToTensor()])

train_dir = ImageFolder(train_dir,transform = transform)

#test_dir = ImageFolder(test_dir,transform = transform)

img,label = train_dir[0]



