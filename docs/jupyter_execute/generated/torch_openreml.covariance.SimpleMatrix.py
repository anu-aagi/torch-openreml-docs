#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import SimpleMatrix

def my_v(free_params):
    n = free_params.shape[0]
    return torch.diag(free_params)

mat = SimpleMatrix(n=3, call=my_v)
mat(torch.tensor([1.0, 2.0, 3.0]))


# In[2]:


mat.grad(torch.tensor([1.0, 2.0, 3.0]))

