#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import EquicorrelationMatrix, HadamardProduct

n = 4
op = HadamardProduct(a=EquicorrelationMatrix(n), b=torch.tensor([5.0]))
free_params = torch.tensor([1.0])
op(free_params)


# In[2]:


import torch
from torch_openreml.covariance import EquicorrelationMatrix, HadamardProduct

op = HadamardProduct(
    a=EquicorrelationMatrix(4),
    b=torch.tensor([5.0])
)
free_params = torch.tensor([1.0])
grad, grad_names = op.manual_grad(free_params)
grad


# In[3]:


grad_names

