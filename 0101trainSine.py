import torch
import math
import numpy as np
import matplotlib.pyplot as plt

###########################
# Step 1: Generate Data
###########################
SAMPLES = 1000
SEED = 1337

np.random.seed(SEED)
torch.manual_seed(SEED)

x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)
np.random.shuffle(x_values)

y_values = np.sin(x_values)
y_values += 0.1 * np.random.randn(*y_values.shape)

# plt.plot(x_values, y_values, 'b.')
# plt.show()

###########################
# Step 2: Split Data
###########################
TRAIN_SPLIT = int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 *SAMPLES + TRAIN_SPLIT)

x_train, x_valid, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_valid, y_test = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

assert (x_train.size + x_valid.size + x_test.size) == SAMPLES

# plt.plot(x_train, y_train, 'bo', label='train')
# plt.plot(x_valid, y_valid, 'yo', label='valid')
# plt.plot(x_test, y_test, 'ro', label='test')

# plt.legend()
# plt.show()

###########################
# Step 3: Build Model
###########################
import torch.nn as nn
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1,16)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(16,16)),    
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(16,1)),    
]))
# print(list(model.parameters()))
# x = torch.tensor([2],dtype=torch.float32)
# ret = model(torch.unsqueeze(x,1))
# print(ret)


x_train = torch.from_numpy(x_train).type(torch.float32).unsqueeze(1)
y_train = torch.from_numpy(y_train).type(torch.float32).unsqueeze(1)


criterion = nn.MSELoss()

# model.zero_grad()
# print('linear.bias.grad before backward')
# print(model.fc1.bias.grad)
# loss.backward()
# print('linear.bias.grad after backward')
# print(model.fc1.bias.grad)

import torch.optim as optim

optimizer = optim.RMSprop(model.parameters(),lr=0.01)


Epochs = 1000
for _ in range(Epochs):

    optimizer.zero_grad()
    ret_train = model(x_train)
    loss = criterion(ret_train, y_train)
    loss.backward()
    optimizer.step()
    
    # print("loss : %.3f" %loss.item())

torch.save(model.state_dict(), 'model/sine_mlp3.pth')

# i = 0.3
# x = torch.tensor([i],dtype=torch.float32)
# print(model(x).item())
# print(np.sin(i))

x_valid = torch.from_numpy(x_valid).type(torch.float32).unsqueeze(1)
y_valid = torch.from_numpy(y_valid).type(torch.float32).unsqueeze(1)

ret_valid = model(x_valid)
plt.plot(x_valid,y_valid,'b.')
plt.plot(x_valid,ret_valid.detach().numpy(),'r.')
plt.show()

