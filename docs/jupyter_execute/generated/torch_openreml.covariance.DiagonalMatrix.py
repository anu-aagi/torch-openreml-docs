#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
mat


# In[2]:


free_params = torch.tensor([0.0, 0.5, 1.0])
mat(free_params)


# In[3]:


mat.grad(free_params)

