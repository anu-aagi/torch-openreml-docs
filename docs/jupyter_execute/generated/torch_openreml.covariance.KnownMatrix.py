#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import KnownMatrix

k = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
mat = KnownMatrix(k)
mat()

