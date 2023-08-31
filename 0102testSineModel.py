import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1,16)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(16,16)),    
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(16,1)),    
]))

model.load_state_dict(torch.load('model/sine_mlp3.pth'))
model.eval()

i = 0.3
x = torch.tensor([i],dtype=torch.float32)
print(model(x).item())
print(np.sin(i))