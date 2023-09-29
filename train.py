import numpy as np
import torch
from torch import optim
from PointProcess import PointProcess
from SyntheticData import GenerativeModel

GP = GenerativeModel()
Y = GP.Data()

delta_loss = torch.tensor(10)
model = PointProcess(Y)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
loss = model.LogMarginalLikelihood 
while(delta_loss > 1e-5):
    optimizer.zero_grad()
    l = -1*loss()
        # print('l:%f' % l)
        # This takes the calculate value, and automatically calculates gradients with respect to parameters
    l.backward(retain_graph=True)
        # Optimizer will take the gradients, and then update parameters accordingly
    optimizer.step()
        # Calculate new loss given the parameter update
    l1 = -1*loss().detach()
    delta_loss = torch.abs(l1 - l)
    print('delta_loss:%f' % delta_loss.detach().numpy())   
    
parameters = [param for param in model.parameters()]
print(parameters)
