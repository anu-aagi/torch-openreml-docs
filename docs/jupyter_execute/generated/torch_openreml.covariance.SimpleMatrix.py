#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import SimpleMatrix

def my_v(free_params):
    return torch.diag(free_params)

mat = SimpleMatrix(num_free_params=3, call=my_v, default=1.0)
mat(torch.tensor([1.0, 2.0, 3.0]))


# In[2]:


mat()


# In[3]:


mat({
    "theta_0": torch.tensor([2.0]),
    "theta_1": torch.tensor([3.0]),
    "theta_2": torch.tensor([4.0]),
})


# In[4]:


mat.grad(torch.tensor([1.0, 2.0, 3.0]))

