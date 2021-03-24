import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
import sa_unet
import data_loading
import albumentations as A
from albumentations.pytorch import ToTensor
from torch.utils.data import random_split
import torch.nn.functional as F
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import seaborn as sns
import pandas as pd

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return -torch.log(dice)

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
                     
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth) / (union + smooth)
                
        return IoU

ALPHA = 0.8
GAMMA = 2

class Comb_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Comb_Loss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        f1 = F1(2)
        f1_score = f1(inputs.cpu(),targets.cpu())
        dice = DiceLoss()
        dice_score = dice(inputs,targets)
        iou = IoU()
        iou_value = iou(inputs,targets)
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE
                       
        return f1_score * dice_score + (1 - iou_value) * focal_loss

if torch.cuda.is_available():
    is_cuda = True
    
TRAIN_PATH = 'train_set/'

def get_train_transform():
   return A.Compose(
       [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])

train_dataset = data_loading.medical_img_data(TRAIN_PATH, transform=get_train_transform())

## Split train and validation set of split ratio 0.15
## 85% of images in train and left 15% of data in valid
split_ratio = 0.15
train_size=int(np.round(train_dataset.__len__()*(1 - split_ratio),0))
valid_size=int(np.round(train_dataset.__len__()*split_ratio,0))

train_data, valid_data = random_split(train_dataset, [train_size, valid_size])

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=10, shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=10) 
dataset_sizes = {'train':len(train_loader.dataset),'valid':len(train_loader.dataset)}
dataloaders = {'train':train_loader,'valid':val_loader}



model_ft = sa_unet.Model(img_channels = 3, n_classes = 1)



if torch.cuda.is_available():
    model_ft = model_ft.cuda()

# Loss, IoU and Optimizer
criterion = Comb_Loss()
accuracy_metric = IoU()
optimizer_ft = optim.Adam(model_ft.parameters(),lr = 1e-3)

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()
    
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)  

            running_loss = []
            running_corrects = []
        
            # Iterate over data
            for inputs,labels in dataloaders[phase]:      
                
                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
    
                # zero the parameter gradients
                optimizer.zero_grad()
               
                # forward
                outputs = model(inputs)
                #pred = outputs.view(outputs.numel())
                #actual = labels.view(labels.numel())

                loss = criterion(outputs, labels)
                score = accuracy_metric(outputs,labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # calculate loss and IoU
                running_loss.append(loss.item())
                running_corrects.append(score.item())
             

            epoch_loss = np.mean(running_loss)
            epoch_acc = np.mean(running_corrects)
            
            print('{} Loss: {:.4f} IoU: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_acc)

            # save parameters
            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                counter = 0
            elif phase == 'valid' and epoch_loss > best_loss:
                counter += 1
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, Loss_list, Accuracy_list

model_ft, Loss_list, Accuracy_list = train_model(model_ft, criterion, optimizer_ft, None,
                       num_epochs=200)

torch.save(model_ft, 'model.pth')

plt.title('Validation loss and IoU',)
sns.set_theme(style="darkgrid")
valid_data = pd.DataFrame({'Loss':Loss_list["valid"], 'IoU':Accuracy_list["valid"]})
sns.lineplot(data=valid_data,dashes=False)
plt.ylabel('Value')
plt.xlabel('Epochs')

plt.figure()
plt.title('Training loss and IoU',)
sns.set_theme(style="darkgrid")
valid_data = pd.DataFrame({'Loss':Loss_list["train"],'IoU':Accuracy_list["train"]})
sns.lineplot(data=valid_data,dashes=False)
plt.ylabel('Value')
plt.xlabel('Epochs')



