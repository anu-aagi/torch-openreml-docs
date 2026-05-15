#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import AR1Matrix

mat = AR1Matrix(4)
free_params = torch.tensor([0.5, 0.0])
mat(free_params)

