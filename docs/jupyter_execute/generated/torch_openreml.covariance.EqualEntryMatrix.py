#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import EqualEntryMatrix

mat = EqualEntryMatrix(3, 2)
mat


# In[2]:


mat = EqualEntryMatrix(3)
mat


# In[3]:


free_params = torch.tensor([0.5])
mat(free_params)


# In[4]:


mat.grad(free_params)

