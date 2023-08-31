import tvm
from tvm import relay
from tvm.relay.backend import Executor
from tvm.relay.backend import Runtime
from tvm.driver import tvmc

import torch
import torch.nn as nn
from collections import OrderedDict


# Step 1: Load Model

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1,16)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(16,16)),    
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(16,1)),    
]))

model.load_state_dict(torch.load('model/sine_mlp3.pth'))
model.eval()

# Step 2: Just-in-time compilation
input_shape = [1,1]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

scripted_model.save('model/sine_mlp3_scripted.pth')












