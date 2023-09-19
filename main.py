import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from dataload import *
from model.classifier import Classifier
from model.Image_classify import *

device = torch.device('cuda')

batch_size = 4
val_size = 5
train_size = len(train_dir) - val_size

train_data , val_data = random_split(train_dir,[train_size,val_size])
print(f'Lenght of Train Data:  {len(train_data)}')
print(f'Length of Test data : {len(val_data)}')

train_dl = DataLoader(train_data,batch_size,shuffle=True,pin_memory=True)
val_dl = DataLoader(val_data,batch_size,pin_memory=True)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []

    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):

        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history


num_epochs = 30
opt_func = torch.optim.Adam
lr = 0.001
#fitting the model on training data and record the result after each epoch
history = fit(num_epochs, lr, Classifier(), train_dl, val_dl, opt_func)


