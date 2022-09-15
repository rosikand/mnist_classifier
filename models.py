"""
File: models.py
------------------
This file holds the torch.nn modules. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchplate
import torchvision
from torchplate import utils


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28*1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        

class ModelInterface(utils.BaseModelInterface):
    def __init__(self): 
        self.model = MLP()
        
        super().__init__(
            model = self.model
        )

    def preprocess(self, img):
        if not torch.is_tensor(img):
            img = torch.tensor(img)
        img = img.float()
        if len(img.shape) == 2:
            img = torch.unsqueeze(img, dim=0)
            img = torch.unsqueeze(img, dim=0)
        if len(img.shape) == 3:
            # remove channel and change to CHW if not already 
            if img.shape[-1] == 1 or img.shape[-1] == 3:
                img = torch.movedim(img, -1, 0)
            img = img[0]
            img = torch.unsqueeze(img, dim=0)
            img = torch.unsqueeze(img, dim=0)
    
        if img.shape[-1] != 28 or img.shape[-2] != 28:
            img = torchvision.transforms.Resize(size=(28, 28))(img)

        
        # normalize if needed 
        if torch.max(img) > 1.1:
            print("YIP")
            img = img/torch.max(img) 

    
        return img

    def predict(self, x):
        logits = self.forward_pipeline(x)
        pred = torch.argmax(F.softmax(logits, dim=1)).item()
        return pred
