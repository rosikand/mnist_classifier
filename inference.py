"""
File: inference.py
------------------
Loads in model and predicts. 
"""

import models 
import torch 
from rsbox import ml

def main():
    net = models.ModelInterface()
    net.load_weights("saved/trained_weights.pth")
    x = torch.randn(28, 28, 1)
    pred = net.predict(x)
    print(pred)

if __name__ == '__main__':
    main()
