import torch.nn as nn
import torch.nn.functional as F
import torch

class ImageclassificationBase(nn.Module):

    def training_step(self,batch):
        images , labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        return loss

    def accuracy(self,outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def validation_step(self,batch):
        images , labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        acc = self.accuracy(out,labels)
        return {'val_loss':loss.detach(),'val_acc':acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
